"""Upstream Learner - The brain that learns which strategies work.

This is the real learning layer that sits OUTSIDE the kernel.
It:
- Selects strategy arms via Thompson Sampling
- Records outcomes to OutcomeDB
- Builds context-aware prompts with strategy + self-critique
- Retrieves similar past failures for context

INVARIANTS:
- Never modifies the kernel or gate
- Only influences: arm selection, prompt context, budgets (within envelope)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rfsn_upstream.bandit import ThompsonBandit
from rfsn_upstream.outcome_db import OutcomeDB, EpisodeRecord
from rfsn_upstream.strategies import SWE_BENCH_ARMS, StrategyArm, get_arm, list_arm_ids
from rfsn_upstream.features import repo_id_from_path


# Self-critique rubric that gets injected into every prompt
SELF_CRITIQUE_RUBRIC = """
## Self-Critique Checklist (MUST complete before proposing any patch)

Before proposing a patch, answer these questions:
1. Did I cite the exact failing assertion/traceback line?
2. Does the patch address that line directly?
3. Is the change minimal and localized?
4. Did I avoid changing unrelated files?
5. Did I introduce broad exception handling or silent failure? (If yes, REMOVE IT.)
6. If I changed an API, did I update call sites or tests accordingly?
7. Am I confident this fixes the root cause, not just a symptom?

If any answer is NO, revise your approach before proposing.
"""


@dataclass(frozen=True)
class LearnerDecision:
    """The learner's decision: which arm to use and how."""
    arm_id: str
    arm: StrategyArm
    similar_failures: list[EpisodeRecord] = field(default_factory=list)
    context_notes: str = ""


class UpstreamLearner:
    """
    SWE-bench upstream learner.
    
    This is the "brain" that:
    - Learns which strategy arms work via Thompson Sampling
    - Records outcomes for future retrieval
    - Provides context from similar past failures
    """

    def __init__(
        self,
        *,
        bandit_db: Path | str,
        outcomes_db: Path | str,
    ):
        self.bandit = ThompsonBandit(
            Path(bandit_db),
            arms=list_arm_ids(),
        )
        self.outcomes = OutcomeDB(outcomes_db)

    def select(
        self,
        *,
        repo_path: Path | str,
        current_fingerprints: list[dict[str, Any]] | None = None,
    ) -> LearnerDecision:
        """
        Select a strategy arm.
        
        Uses Thompson Sampling, optionally penalizing arms that 
        recently failed on similar fingerprints.
        """
        # Get arm IDs
        arm_ids = list_arm_ids()
        
        # Basic Thompson selection
        arm_id = self.bandit.select_arm()
        arm = get_arm(arm_id)
        
        # Look for similar past failures (if fingerprints provided)
        similar_failures: list[EpisodeRecord] = []
        context_notes = ""
        
        if current_fingerprints:
            for fp in current_fingerprints[:3]:  # Limit to first 3
                fp_id = fp.get("fingerprint_id", "")
                if fp_id:
                    episodes = self.outcomes.episodes_with_fingerprint(fp_id, limit=5)
                    similar_failures.extend(episodes)
            
            # Deduplicate
            seen = set()
            unique_failures = []
            for ep in similar_failures:
                key = (ep.instance_id, ep.arm_id)
                if key not in seen:
                    seen.add(key)
                    unique_failures.append(ep)
            similar_failures = unique_failures[:10]
            
            # Build context notes from successful similar repairs
            successful = [ep for ep in similar_failures if ep.success]
            if successful:
                context_notes = self._build_context_notes(successful)
        
        return LearnerDecision(
            arm_id=arm_id,
            arm=arm,
            similar_failures=similar_failures,
            context_notes=context_notes,
        )

    def _build_context_notes(self, successful_episodes: list[EpisodeRecord]) -> str:
        """Build context notes from successful similar repairs."""
        if not successful_episodes:
            return ""
        
        notes = ["## Similar Past Successes\n"]
        for i, ep in enumerate(successful_episodes[:3], 1):
            notes.append(f"{i}. Instance `{ep.instance_id}` fixed with strategy `{ep.arm_id}`")
            if ep.metrics.get("completion_reason"):
                notes.append(f"   - Completed: {ep.metrics['completion_reason']}")
        
        return "\n".join(notes)

    def build_system_addendum(self, decision: LearnerDecision) -> str:
        """
        Build the system prompt addendum that gets injected into the planner.
        
        This is the primary interface between upstream learning and the kernel.
        """
        sections = [
            "---",
            "## SWE-bench Repair Strategy",
            f"**Active Arm**: `{decision.arm_id}` ({decision.arm.name})",
            f"**Patch Style**: {decision.arm.patch_style}",
            "",
            "### Planning Steps",
            decision.arm.planning_instructions,
            "",
            SELF_CRITIQUE_RUBRIC,
        ]
        
        if decision.context_notes:
            sections.append("")
            sections.append(decision.context_notes)
        
        # Warn about similar failures
        failed_arms = set()
        for ep in decision.similar_failures:
            if not ep.success:
                failed_arms.add(ep.arm_id)
        
        if failed_arms:
            sections.append("")
            sections.append("## Warning: These strategies failed on similar errors")
            sections.append(", ".join(f"`{a}`" for a in failed_arms))
            sections.append("Consider a different approach if using one of these.")
        
        return "\n".join(sections)

    def update(
        self,
        *,
        instance_id: str,
        repo_path: Path | str,
        arm_id: str,
        success: bool,
        metrics: dict[str, Any],
        fingerprints: list[dict[str, Any]],
    ) -> int:
        """
        Record an episode outcome.
        
        This is called AFTER the kernel completes (success or failure).
        Updates both the outcome DB and the bandit posterior.
        """
        repo_id = repo_id_from_path(repo_path)
        
        # Record to outcome DB
        episode_id = self.outcomes.add_episode(
            instance_id=instance_id,
            repo_id=repo_id,
            arm_id=arm_id,
            success=success,
            metrics=metrics,
            fingerprints=fingerprints,
        )
        
        # Update bandit
        self.bandit.update(arm_id, success=success)
        
        return episode_id

    def get_arm_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all arms."""
        stats = {}
        for arm_id in list_arm_ids():
            arm_stats = self.bandit.get_arm_stats(arm_id)
            db_success, db_total = self.outcomes.arm_success_rate(arm_id)
            stats[arm_id] = {
                "bandit_pulls": arm_stats.pulls,
                "bandit_successes": arm_stats.successes,
                "db_total": db_total,
                "db_successes": db_success,
                "success_rate": db_success / db_total if db_total > 0 else 0.0,
            }
        return stats


def create_learner_state(dir_path: Path | str) -> tuple[str, str]:
    """
    Helper for CI: creates default database locations.
    
    Returns (bandit_db_path, outcomes_db_path).
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    bandit_db = str(d / "bandit.db")
    outcomes_db = str(d / "outcomes.db")
    
    # Initialize DBs
    ThompsonBandit(Path(bandit_db), arms=list_arm_ids())
    OutcomeDB(outcomes_db)
    
    return bandit_db, outcomes_db
