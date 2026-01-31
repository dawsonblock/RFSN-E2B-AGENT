"""RFSN Deterministic Agent Kernel.

A safety-first kernel for autonomous code repair that enforces:
- Serial authority (one action at a time)
- Separated planning, gating, and execution
- Deterministic, immutable gate
- Learning only upstream

Non-Negotiable Invariants:
1. Planner never executes
2. Gate never learns
3. Controller never decides
4. All commits are serial
5. No hidden state across decisions
6. All side effects are logged
7. Rejected actions produce evidence
"""

from .state import StateSnapshot, TestStatus, TaskProgress, SafetyEnvelope
from .proposal import Proposal, ProposalIntent
from .gate import Gate, GateDecision, GateConfig
from .controller import Controller, ControllerConfig, ExecutionResult
from .ledger import Ledger, LedgerEntry
from .kernel import run_kernel, Kernel, KernelConfig, KernelResult
from .planner_interface import Planner, NullPlanner, StaticPlanner, SequentialPlanner

__version__ = "1.0.0"
__all__ = [
    # State
    "StateSnapshot",
    "TestStatus",
    "TaskProgress",
    "SafetyEnvelope",
    # Proposal
    "Proposal",
    "ProposalIntent",
    # Gate
    "Gate",
    "GateDecision",
    "GateConfig",
    # Controller
    "Controller",
    "ControllerConfig",
    "ExecutionResult",
    # Ledger
    "Ledger",
    "LedgerEntry",
    # Kernel
    "run_kernel",
    "Kernel",
    "KernelConfig",
    "KernelResult",
    # Planner
    "Planner",
    "NullPlanner",
    "StaticPlanner",
    "SequentialPlanner",
]
