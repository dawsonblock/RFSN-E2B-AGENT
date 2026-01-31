"""Kernel - Main Orchestration Loop.

The Kernel orchestrates the RFSN loop:
    proposal → gate → decision → (execute → ledger) OR (record_rejection)

INVARIANTS:
1. Planner never executes
2. Gate never learns
3. Controller never decides
4. All commits are serial
5. No hidden state across decisions
6. All side effects are logged
7. Rejected actions produce evidence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controller import Controller, ControllerConfig, ExecutionResult
from .gate import Gate, GateConfig, GateDecision
from .ledger import Ledger, create_ledger
from .planner_interface import NullPlanner, Planner
from .proposal import Proposal
from .state import (
    SafetyEnvelope,
    StateSnapshot,
    TaskProgress,
    TestResult,
    TestStatus,
    create_initial_state,
    update_state,
)

logger = logging.getLogger(__name__)


@dataclass
class KernelConfig:
    """Configuration for the RFSN Kernel."""
    
    # Component configs
    gate_config: GateConfig = field(default_factory=GateConfig)
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    safety_envelope: SafetyEnvelope = field(default_factory=SafetyEnvelope)
    
    # Execution limits
    max_steps: int = 20
    max_rejections_per_step: int = 5
    max_total_rejections: int = 50
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"
    
    # Behavior
    stop_on_success: bool = True
    require_test_pass: bool = True


@dataclass
class KernelResult:
    """Result of running the kernel on a task."""
    
    success: bool
    task_id: str
    final_state: StateSnapshot
    completion_reason: str
    total_steps: int
    total_proposals: int
    accepted_proposals: int
    rejected_proposals: int
    test_status: str
    duration_seconds: float
    ledger_path: str
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "task_id": self.task_id,
            "final_state_hash": self.final_state.state_hash,
            "completion_reason": self.completion_reason,
            "total_steps": self.total_steps,
            "total_proposals": self.total_proposals,
            "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejected_proposals,
            "test_status": self.test_status,
            "duration_seconds": self.duration_seconds,
            "ledger_path": self.ledger_path,
        }


def run_kernel(
    task_id: str,
    task: str,
    workspace_path: Path | str,
    planner: Planner,
    config: KernelConfig | None = None,
    ledger: Ledger | None = None,
) -> KernelResult:
    """Run the RFSN kernel on a task.
    
    This is the main entry point for the kernel.
    
    The execution loop is:
        1. Planner proposes action
        2. Gate validates proposal
        3. If accepted: Controller executes, Ledger records
        4. If rejected: Ledger records rejection, Planner observes
        5. Update state
        6. Repeat until complete or limits reached
    
    INVARIANTS ENFORCED:
    1. Planner never executes
    2. Gate never learns
    3. Controller never decides
    4. All commits are serial
    5. No hidden state across decisions
    6. All side effects are logged
    7. Rejected actions produce evidence
    
    Args:
        task_id: Unique identifier for this task.
        task: Task description for the planner.
        workspace_path: Path to the workspace.
        planner: Planner instance to use.
        config: Kernel configuration.
        ledger: Optional ledger (creates new one if not provided).
    
    Returns:
        KernelResult with success status and statistics.
    """
    start_time = time.perf_counter()
    config = config or KernelConfig()
    workspace_path = Path(workspace_path).resolve()
    
    # Setup logging
    if config.verbose:
        logging.basicConfig(level=getattr(logging, config.log_level))
    
    # Initialize components
    gate = Gate(config.gate_config)
    controller = Controller(config.controller_config)
    ledger = ledger or create_ledger(task_id, workspace_path)
    
    # Initialize state
    state = create_initial_state(
        task_id=task_id,
        workspace_root=workspace_path,
        safety_envelope=config.safety_envelope,
    )
    
    # Record initial state
    ledger.record_state(state, "Initial state")
    
    # Tracking
    total_proposals = 0
    accepted_proposals = 0
    rejected_proposals = 0
    rejections_this_step = 0
    
    logger.info(f"Starting kernel for task: {task_id}")
    logger.info(f"Workspace: {workspace_path}")
    logger.info(f"Planner: {planner.get_name()}")
    
    completion_reason = "unknown"
    
    # Main loop
    while True:
        step_index = state.task_progress.step_index
        
        # Check step limit
        if step_index >= config.max_steps:
            completion_reason = f"Max steps reached: {config.max_steps}"
            logger.info(completion_reason)
            break
        
        # Check rejection limits
        if rejected_proposals >= config.max_total_rejections:
            completion_reason = f"Max rejections reached: {config.max_total_rejections}"
            logger.info(completion_reason)
            break
        
        if rejections_this_step >= config.max_rejections_per_step:
            completion_reason = f"Max rejections per step: {config.max_rejections_per_step}"
            logger.info(completion_reason)
            break
        
        # 1. PLANNER PROPOSES (never executes)
        context = {
            "step_index": step_index,
            "rejected_this_step": rejections_this_step,
            "test_status": state.test_result.status.value,
        }
        
        proposal = planner.propose(state, task, context)
        
        if proposal is None:
            # Planner has no more proposals
            completion_reason = "Planner returned None (task complete or no ideas)"
            logger.info(completion_reason)
            break
        
        total_proposals += 1
        logger.info(f"Step {step_index}: Proposal {proposal.proposal_id} ({proposal.intent.value})")
        
        # 2. GATE VALIDATES (never learns)
        decision = gate.validate(state, proposal)
        
        logger.info(f"  Gate decision: {'ACCEPT' if decision.accepted else 'REJECT'}")
        if not decision.accepted:
            logger.info(f"  Reason: {decision.reason}")
        
        # 3. DISPATCH BASED ON DECISION
        if decision.accepted:
            # 3a. CONTROLLER EXECUTES (never decides)
            result = controller.execute(proposal, workspace_path, gate_approved=True)
            
            # Record in ledger
            ledger.record(state, proposal, result)
            
            # Notify planner of success
            planner.observe_success(proposal, result)
            
            # Update state
            new_progress = TaskProgress(
                step_index=step_index + 1,
                total_proposals=total_proposals,
                accepted_proposals=accepted_proposals + 1,
                rejected_proposals=rejected_proposals,
                patches_applied=state.task_progress.patches_applied + (
                    1 if proposal.intent.value in ("modify_file", "create_file") else 0
                ),
                tests_run=state.task_progress.tests_run + (
                    1 if proposal.intent.value in ("run_test", "run_focused_test") else 0
                ),
            )
            
            # Update test result if this was a test run
            new_test_result = state.test_result
            if proposal.intent.value in ("run_test", "run_focused_test"):
                test_passed = dict(result.artifacts).get("tests_passed", False)
                new_test_result = TestResult(
                    status=TestStatus.PASSING if test_passed else TestStatus.FAILING,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            
            state = update_state(
                state,
                task_progress=new_progress,
                test_result=new_test_result,
            )
            
            accepted_proposals += 1
            rejections_this_step = 0  # Reset for new step
            
            logger.info(f"  Execution: {'SUCCESS' if result.success else 'FAILED'}")
            
            # Check for task completion
            if config.stop_on_success and config.require_test_pass:
                if new_test_result.status == TestStatus.PASSING:
                    completion_reason = "Tests passing"
                    logger.info(completion_reason)
                    break
            
        else:
            # 3b. RECORD REJECTION (rejected actions produce evidence)
            ledger.record_rejection(state, proposal, decision)
            
            # Notify planner (for upstream learning)
            planner.observe_rejection(proposal, decision)
            
            rejected_proposals += 1
            rejections_this_step += 1
            
            # Update state to record rejection
            new_progress = TaskProgress(
                step_index=step_index,  # Don't increment step for rejection
                total_proposals=total_proposals,
                accepted_proposals=accepted_proposals,
                rejected_proposals=rejected_proposals,
            )
            
            state = update_state(state, task_progress=new_progress)
    
    # Record completion
    success = (
        state.test_result.status == TestStatus.PASSING
        if config.require_test_pass
        else accepted_proposals > 0
    )
    
    ledger.record_completion(state, success, completion_reason)
    
    duration = time.perf_counter() - start_time
    
    logger.info(f"Kernel completed: success={success}, reason={completion_reason}")
    logger.info(f"Stats: {accepted_proposals} accepted, {rejected_proposals} rejected, {duration:.2f}s")
    
    return KernelResult(
        success=success,
        task_id=task_id,
        final_state=state,
        completion_reason=completion_reason,
        total_steps=state.task_progress.step_index,
        total_proposals=total_proposals,
        accepted_proposals=accepted_proposals,
        rejected_proposals=rejected_proposals,
        test_status=state.test_result.status.value,
        duration_seconds=duration,
        ledger_path=str(ledger.path),
    )


class Kernel:
    """Object-oriented wrapper for the kernel.
    
    Provides a stateful interface to the kernel for more complex use cases.
    
    Usage:
        kernel = Kernel(config)
        result = kernel.run(task_id, task, workspace, planner)
    """
    
    def __init__(self, config: KernelConfig | None = None):
        """Initialize kernel with config.
        
        Args:
            config: Kernel configuration.
        """
        self.config = config or KernelConfig()
        self._run_count = 0
    
    def run(
        self,
        task_id: str,
        task: str,
        workspace_path: Path | str,
        planner: Planner,
        ledger: Ledger | None = None,
    ) -> KernelResult:
        """Run the kernel on a task.
        
        Args:
            task_id: Unique task identifier.
            task: Task description.
            workspace_path: Workspace path.
            planner: Planner instance.
            ledger: Optional ledger.
        
        Returns:
            KernelResult.
        """
        self._run_count += 1
        return run_kernel(
            task_id=task_id,
            task=task,
            workspace_path=workspace_path,
            planner=planner,
            config=self.config,
            ledger=ledger,
        )
    
    @property
    def run_count(self) -> int:
        """Get number of runs completed."""
        return self._run_count
