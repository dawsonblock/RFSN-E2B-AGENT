"""Planner Interface - Abstract protocol for planners.

Planners are NON-TRUSTED components that generate proposals.
They NEVER execute actions directly.

INVARIANTS:
1. Planner has READ-ONLY access to state
2. Planner produces PROPOSALS, never executes
3. Planner receives REJECTION FEEDBACK (for learning upstream)
4. Planners are REPLACEABLE (protocol-based)
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gate import GateDecision
    from .proposal import Proposal
    from .state import StateSnapshot


class Planner(abc.ABC):
    """Abstract planner protocol.
    
    Planners are non-trusted components that propose actions.
    They are given:
    - Read-only state
    - Task description
    - History of rejections (for learning)
    
    They produce:
    - Proposals (structured action descriptions)
    
    They NEVER:
    - Execute actions
    - Modify state directly
    - Bypass the gate
    
    INVARIANTS:
    1. propose() returns a Proposal or None
    2. observe_rejection() is for feedback only
    3. Planners can be swapped without changing the kernel
    
    Usage:
        class MyPlanner(Planner):
            def propose(self, state, task) -> Proposal | None:
                # Generate proposal from state and task
                return create_proposal(...)
            
            def observe_rejection(self, proposal, decision):
                # Learn from rejection (optional)
                pass
    """
    
    @abc.abstractmethod
    def propose(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Proposal | None:
        """Generate a proposal for the current state.
        
        Args:
            state: Current state snapshot (READ-ONLY).
            task: Task description.
            context: Optional context (e.g., history, hints).
        
        Returns:
            A Proposal if the planner has something to suggest,
            None if the planner thinks the task is complete or has no ideas.
        """
        ...
    
    @abc.abstractmethod
    def observe_rejection(
        self,
        proposal: Proposal,
        decision: GateDecision,
    ) -> None:
        """Observe a rejection for feedback.
        
        This method is called when a proposal is rejected by the gate.
        The planner can use this for:
        - Updating internal state (if stateful)
        - Generating better proposals next time
        - Collecting feedback for upstream learning
        
        INVARIANT: This method must NOT modify global state or bypass the gate.
        
        Args:
            proposal: The rejected proposal.
            decision: The gate's decision with reason.
        """
        ...
    
    def observe_success(
        self,
        proposal: Proposal,
        result: Any,
    ) -> None:
        """Observe a successful execution for feedback.
        
        This method is called when a proposal is accepted and executed.
        Optional to implement.
        
        Args:
            proposal: The executed proposal.
            result: The execution result.
        """
        pass
    
    def get_name(self) -> str:
        """Get the planner's name for logging."""
        return self.__class__.__name__
    
    def get_config(self) -> dict[str, Any]:
        """Get the planner's configuration for audit logging."""
        return {"name": self.get_name()}


class NullPlanner(Planner):
    """A planner that always returns None.
    
    Useful for testing the kernel without a real planner.
    """
    
    def propose(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Proposal | None:
        """Always returns None (no proposals)."""
        return None
    
    def observe_rejection(
        self,
        proposal: Proposal,
        decision: GateDecision,
    ) -> None:
        """No-op for null planner."""
        pass


class StaticPlanner(Planner):
    """A planner that returns proposals from a predefined list.
    
    Useful for testing and replay scenarios.
    """
    
    def __init__(self, proposals: list[Proposal]):
        """Initialize with list of proposals to return.
        
        Args:
            proposals: List of proposals to return in order.
        """
        self._proposals = list(proposals)
        self._index = 0
        self._rejections: list[tuple[Proposal, GateDecision]] = []
    
    def propose(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Proposal | None:
        """Return next proposal from list."""
        if self._index < len(self._proposals):
            proposal = self._proposals[self._index]
            self._index += 1
            return proposal
        return None
    
    def observe_rejection(
        self,
        proposal: Proposal,
        decision: GateDecision,
    ) -> None:
        """Record rejection for inspection."""
        self._rejections.append((proposal, decision))
    
    def get_rejections(self) -> list[tuple[Proposal, GateDecision]]:
        """Get all recorded rejections."""
        return list(self._rejections)


class SequentialPlanner(Planner):
    """A planner that tries multiple strategies in sequence.
    
    If one strategy fails (proposal rejected), tries the next.
    """
    
    def __init__(self, planners: list[Planner]):
        """Initialize with list of planners to try.
        
        Args:
            planners: List of planners to try in order.
        """
        self._planners = planners
        self._current_index = 0
        self._last_rejected_by: dict[str, int] = {}  # proposal_id -> planner_index
    
    def propose(
        self,
        state: StateSnapshot,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Proposal | None:
        """Try planners in sequence until one produces a proposal."""
        for i in range(self._current_index, len(self._planners)):
            planner = self._planners[i]
            proposal = planner.propose(state, task, context)
            if proposal is not None:
                self._last_rejected_by[proposal.proposal_id] = i
                return proposal
        return None
    
    def observe_rejection(
        self,
        proposal: Proposal,
        decision: GateDecision,
    ) -> None:
        """Forward rejection to appropriate planner and try next."""
        if proposal.proposal_id in self._last_rejected_by:
            idx = self._last_rejected_by[proposal.proposal_id]
            self._planners[idx].observe_rejection(proposal, decision)
            # Move to next planner
            self._current_index = idx + 1
