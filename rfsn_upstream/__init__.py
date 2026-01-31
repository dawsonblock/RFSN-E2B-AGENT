"""RFSN Upstream Learner.

The upstream learner lives OUTSIDE the kernel and:
- Selects strategy arms via Thompson Sampling (bandit)
- Records outcomes to persistent database
- Fingerprints failures for retrieval and "don't repeat" logic
- Builds context-aware prompts with strategy + self-critique

INVARIANTS:
1. Learner NEVER modifies kernel decisions
2. Learner ONLY affects proposal generation (coaching layer)
3. Learner state is persisted for CI/CD integration
4. All learning happens OUTSIDE the kernel execution loop
"""

# New real learner components
from .outcome_db import OutcomeDB, EpisodeRecord
from .strategies import SWE_BENCH_ARMS, StrategyArm, get_arm, list_arm_ids
from .features import (
    repo_id_from_path,
    summarize_test_failure,
    fingerprints_from_test,
    fingerprints_from_gate_rejection,
    compute_diff_fingerprint,
)
from .upstream_learner import (
    UpstreamLearner,
    LearnerDecision,
    create_learner_state,
    SELF_CRITIQUE_RUBRIC,
)

__version__ = "2.0.0"
__all__ = [
    # Outcome DB
    "OutcomeDB",
    "EpisodeRecord",
    # Strategies
    "SWE_BENCH_ARMS",
    "StrategyArm",
    "get_arm",
    "list_arm_ids",
    # Features
    "repo_id_from_path",
    "summarize_test_failure",
    "fingerprints_from_test",
    "fingerprints_from_gate_rejection",
    "compute_diff_fingerprint",
    # Upstream Learner
    "UpstreamLearner",
    "LearnerDecision",
    "create_learner_state",
    "SELF_CRITIQUE_RUBRIC",
]
