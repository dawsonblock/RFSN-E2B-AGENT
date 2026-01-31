"""RFSN Upstream Learner.

The upstream learner lives OUTSIDE the kernel and:
- Selects prompt variants (bandit)
- Fingerprints failures for retrieval
- Stores memories for context
- Updates based on task outcomes

INVARIANTS:
1. Learner NEVER modifies kernel decisions
2. Learner ONLY affects proposal generation (coaching layer)
3. Learner state is persisted for CI/CD integration
"""

from .bandit import ThompsonBandit, ArmStats
from .fingerprint import Fingerprint, compute_fingerprint, fingerprint_from_rejection
from .retrieval import Memory, MemoryIndex
from .prompt_variants import PROMPT_VARIANTS, get_variant, PromptVariant

__version__ = "1.0.0"
__all__ = [
    # Bandit
    "ThompsonBandit",
    "ArmStats",
    # Fingerprinting
    "Fingerprint",
    "compute_fingerprint",
    "fingerprint_from_rejection",
    # Retrieval
    "Memory",
    "MemoryIndex",
    # Prompts
    "PROMPT_VARIANTS",
    "get_variant",
    "PromptVariant",
]
