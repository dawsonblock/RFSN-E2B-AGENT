"""Tests for RFSN Upstream Learner components.

Tests:
- OutcomeDB persistence and queries
- StrategyArm definitions and access
- Feature extraction and fingerprinting
- UpstreamLearner integration
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest


# --- Mock classes for testing ---

class MockTestResult:
    """Mock test result for feature extraction testing."""
    
    def __init__(
        self,
        status: str = "failed",
        stdout: str = "",
        stderr: str = "",
        passed: int = 0,
        failed: int = 1,
        errors: int = 0,
        skipped: int = 0,
        duration_seconds: float = 1.0,
        failing_tests: Optional[List[str]] = None,
    ):
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.passed = passed
        self.failed = failed
        self.errors = errors
        self.skipped = skipped
        self.duration_seconds = duration_seconds
        self.failing_tests = failing_tests or []


# =============================================================================
# OutcomeDB Tests
# =============================================================================

class TestOutcomeDB:
    """Tests for OutcomeDB persistence."""
    
    def test_create_empty_db(self, tmp_path: Path):
        """DB can be created fresh."""
        from rfsn_upstream import OutcomeDB
        
        db = OutcomeDB(tmp_path / "test.db")
        assert db.total_episodes() == 0
    
    def test_add_episode(self, tmp_path: Path):
        """Episodes can be added and retrieved."""
        from rfsn_upstream import OutcomeDB
        
        db = OutcomeDB(tmp_path / "test.db")
        
        episode_id = db.add_episode(
            instance_id="django__django-12345",
            repo_id="abc123",
            arm_id="swe_minimal_diff",
            success=True,
            metrics={"steps": 5, "duration": 30.0},
            fingerprints=[{"fingerprint_id": "fp123", "category": "test"}],
        )
        
        assert episode_id == 1
        assert db.total_episodes() == 1
    
    def test_recent_episodes(self, tmp_path: Path):
        """Can query recent episodes."""
        from rfsn_upstream import OutcomeDB
        
        db = OutcomeDB(tmp_path / "test.db")
        
        # Add multiple episodes
        for i in range(5):
            db.add_episode(
                instance_id=f"test_instance_{i}",
                repo_id="repo123",
                arm_id="swe_minimal_diff" if i % 2 == 0 else "swe_traceback_first",
                success=i % 2 == 0,
                metrics={"step": i},
                fingerprints=[],
            )
        
        # Query all
        all_eps = db.recent_episodes(limit=10)
        assert len(all_eps) == 5
        
        # Query by success
        success_eps = db.recent_episodes(success=True, limit=10)
        assert len(success_eps) == 3
        
        # Query by arm
        arm_eps = db.recent_episodes(arm_id="swe_traceback_first", limit=10)
        assert len(arm_eps) == 2
    
    def test_arm_success_rate(self, tmp_path: Path):
        """Can calculate success rate per arm."""
        from rfsn_upstream import OutcomeDB
        
        db = OutcomeDB(tmp_path / "test.db")
        
        # Add 10 episodes: 7 success, 3 failure
        for i in range(10):
            db.add_episode(
                instance_id=f"test_{i}",
                repo_id="repo",
                arm_id="test_arm",
                success=i < 7,
                metrics={},
                fingerprints=[],
            )
        
        successes, total = db.arm_success_rate("test_arm")
        assert successes == 7
        assert total == 10
    
    def test_episodes_with_fingerprint(self, tmp_path: Path):
        """Can find episodes by fingerprint."""
        from rfsn_upstream import OutcomeDB
        
        db = OutcomeDB(tmp_path / "test.db")
        
        # Add episodes with different fingerprints
        db.add_episode(
            instance_id="ep1",
            repo_id="repo",
            arm_id="arm1",
            success=True,
            metrics={},
            fingerprints=[{"fingerprint_id": "fp_shared", "category": "test"}],
        )
        db.add_episode(
            instance_id="ep2",
            repo_id="repo",
            arm_id="arm2",
            success=False,
            metrics={},
            fingerprints=[{"fingerprint_id": "fp_shared", "category": "test"}],
        )
        db.add_episode(
            instance_id="ep3",
            repo_id="repo",
            arm_id="arm1",
            success=True,
            metrics={},
            fingerprints=[{"fingerprint_id": "fp_unique", "category": "test"}],
        )
        
        # Find by shared fingerprint
        shared = db.episodes_with_fingerprint("fp_shared", limit=10)
        assert len(shared) == 2
        
        # Find by unique fingerprint
        unique = db.episodes_with_fingerprint("fp_unique", limit=10)
        assert len(unique) == 1
        assert unique[0].instance_id == "ep3"
    
    def test_persistence(self, tmp_path: Path):
        """DB persists across instances."""
        from rfsn_upstream import OutcomeDB
        
        db_path = tmp_path / "persist.db"
        
        # Create and add
        db1 = OutcomeDB(db_path)
        db1.add_episode(
            instance_id="persist_test",
            repo_id="repo",
            arm_id="arm",
            success=True,
            metrics={"test": True},
            fingerprints=[],
        )
        
        # Create new instance
        db2 = OutcomeDB(db_path)
        episodes = db2.recent_episodes(limit=10)
        
        assert len(episodes) == 1
        assert episodes[0].instance_id == "persist_test"


# =============================================================================
# Strategy Arm Tests
# =============================================================================

class TestStrategyArms:
    """Tests for strategy arm definitions."""
    
    def test_list_arm_ids(self):
        """All arms have unique IDs."""
        from rfsn_upstream import list_arm_ids
        
        ids = list_arm_ids()
        assert len(ids) >= 7  # At least 7 arms
        assert len(ids) == len(set(ids))  # All unique
    
    def test_get_arm(self):
        """Can retrieve arm by ID."""
        from rfsn_upstream import get_arm, list_arm_ids
        
        for arm_id in list_arm_ids():
            arm = get_arm(arm_id)
            assert arm.arm_id == arm_id
            assert arm.name
            assert arm.patch_style
            assert arm.planning_instructions
            assert arm.max_steps > 0
            assert arm.max_patches > 0
    
    def test_get_arm_invalid(self):
        """Getting invalid arm raises ValueError."""
        from rfsn_upstream import get_arm
        
        with pytest.raises(ValueError):
            get_arm("nonexistent_arm")
    
    def test_swe_bench_arms_content(self):
        """SWE-bench arms have expected content."""
        from rfsn_upstream import SWE_BENCH_ARMS
        
        # Check for required arms
        arm_names = {arm.name for arm in SWE_BENCH_ARMS}
        assert "Minimal diff" in arm_names
        assert "Traceback-first" in arm_names
        
        # Check structure
        for arm in SWE_BENCH_ARMS:
            assert arm.arm_id.startswith("swe_")
            assert "\n" in arm.planning_instructions  # Multi-step


# =============================================================================
# Feature Extraction Tests
# =============================================================================

class TestFeatureExtraction:
    """Tests for feature extraction and fingerprinting."""
    
    def test_repo_id_from_path(self):
        """Repo ID is deterministic hash of path."""
        from rfsn_upstream import repo_id_from_path
        
        # Same path = same ID
        id1 = repo_id_from_path(Path("/foo/bar"))
        id2 = repo_id_from_path(Path("/foo/bar"))
        assert id1 == id2
        
        # Different path = different ID
        id3 = repo_id_from_path(Path("/foo/baz"))
        assert id1 != id3
    
    def test_summarize_test_failure_none(self):
        """Summarize handles None test result."""
        from rfsn_upstream import summarize_test_failure
        
        summary = summarize_test_failure(None)
        assert summary["has_test"] is False
    
    def test_summarize_test_failure(self):
        """Summarize extracts key fields from test result."""
        from rfsn_upstream import summarize_test_failure
        
        test = MockTestResult(
            status="failed",
            stdout="FAILED test_foo::test_bar - AssertionError",
            stderr="E   AssertionError: 1 != 2",
            passed=5,
            failed=1,
            errors=0,
            failing_tests=["test_foo::test_bar"],
        )
        
        summary = summarize_test_failure(test)
        
        assert summary["has_test"] is True
        assert summary["status"] == "failed"
        assert summary["passed"] == 5
        assert summary["failed"] == 1
        assert "test_foo::test_bar" in summary["failing_tests"]
    
    def test_fingerprints_from_test_none(self):
        """Fingerprints from None returns empty list."""
        from rfsn_upstream import fingerprints_from_test
        
        fps = fingerprints_from_test(None)
        assert fps == []
    
    def test_fingerprints_from_test(self):
        """Fingerprints extracted from test failures."""
        from rfsn_upstream import fingerprints_from_test
        
        test = MockTestResult(
            stdout="FAILED - TypeError: 'NoneType' object is not subscriptable",
            stderr="E   TypeError: 'NoneType' object is not subscriptable",
        )
        
        fps = fingerprints_from_test(test)
        
        # Should have fingerprints
        assert len(fps) > 0
        
        # Each fingerprint has required fields
        for fp in fps:
            assert "fingerprint_id" in fp
            assert "category" in fp
            assert len(fp["fingerprint_id"]) == 16  # SHA256 truncated
    
    def test_fingerprints_deterministic(self):
        """Same test produces same fingerprints."""
        from rfsn_upstream import fingerprints_from_test
        
        test1 = MockTestResult(
            stdout="Error: KeyError: 'missing_key'",
            stderr="KeyError: 'missing_key'",
        )
        test2 = MockTestResult(
            stdout="Error: KeyError: 'missing_key'",
            stderr="KeyError: 'missing_key'",
        )
        
        fps1 = fingerprints_from_test(test1)
        fps2 = fingerprints_from_test(test2)
        
        assert fps1 == fps2


# =============================================================================
# UpstreamLearner Integration Tests
# =============================================================================

class TestUpstreamLearner:
    """Integration tests for UpstreamLearner."""
    
    def test_create_learner(self, tmp_path: Path):
        """Learner can be created fresh."""
        from rfsn_upstream import UpstreamLearner
        
        learner = UpstreamLearner(
            bandit_db=tmp_path / "bandit.db",
            outcomes_db=tmp_path / "outcomes.db",
        )
        
        assert learner.bandit is not None
        assert learner.outcomes is not None
    
    def test_select_arm(self, tmp_path: Path):
        """Learner can select an arm."""
        from rfsn_upstream import UpstreamLearner, list_arm_ids
        
        learner = UpstreamLearner(
            bandit_db=tmp_path / "bandit.db",
            outcomes_db=tmp_path / "outcomes.db",
        )
        
        decision = learner.select(repo_path=tmp_path)
        
        assert decision.arm_id in list_arm_ids()
        assert decision.arm is not None
        assert decision.arm.arm_id == decision.arm_id
    
    def test_build_system_addendum(self, tmp_path: Path):
        """System addendum includes strategy and self-critique."""
        from rfsn_upstream import UpstreamLearner
        
        learner = UpstreamLearner(
            bandit_db=tmp_path / "bandit.db",
            outcomes_db=tmp_path / "outcomes.db",
        )
        
        decision = learner.select(repo_path=tmp_path)
        addendum = learner.build_system_addendum(decision)
        
        # Check required sections
        assert "SWE-bench Repair Strategy" in addendum
        assert decision.arm_id in addendum
        assert decision.arm.name in addendum
        assert "Self-Critique Checklist" in addendum
        assert "minimal and localized" in addendum
    
    def test_update_records_outcome(self, tmp_path: Path):
        """Update records episode to both DBs."""
        from rfsn_upstream import UpstreamLearner
        
        learner = UpstreamLearner(
            bandit_db=tmp_path / "bandit.db",
            outcomes_db=tmp_path / "outcomes.db",
        )
        
        decision = learner.select(repo_path=tmp_path)
        
        # Update with outcome
        episode_id = learner.update(
            instance_id="test_instance",
            repo_path=tmp_path,
            arm_id=decision.arm_id,
            success=True,
            metrics={"steps": 3},
            fingerprints=[{"fingerprint_id": "test_fp", "category": "test"}],
        )
        
        assert episode_id > 0
        assert learner.outcomes.total_episodes() == 1
    
    def test_get_arm_stats(self, tmp_path: Path):
        """Can get stats for all arms."""
        from rfsn_upstream import UpstreamLearner, list_arm_ids
        
        learner = UpstreamLearner(
            bandit_db=tmp_path / "bandit.db",
            outcomes_db=tmp_path / "outcomes.db",
        )
        
        stats = learner.get_arm_stats()
        
        assert len(stats) == len(list_arm_ids())
        for arm_id, s in stats.items():
            assert "bandit_pulls" in s
            assert "bandit_successes" in s
            assert "db_total" in s
    
    def test_similar_failure_retrieval(self, tmp_path: Path):
        """Learner finds similar past failures."""
        from rfsn_upstream import UpstreamLearner
        
        learner = UpstreamLearner(
            bandit_db=tmp_path / "bandit.db",
            outcomes_db=tmp_path / "outcomes.db",
        )
        
        # Add episode with specific fingerprint
        learner.update(
            instance_id="past_failure",
            repo_path=tmp_path,
            arm_id="swe_minimal_diff",
            success=False,
            metrics={},
            fingerprints=[{"fingerprint_id": "shared_fp", "category": "type_error"}],
        )
        
        # Select with same fingerprint
        decision = learner.select(
            repo_path=tmp_path,
            current_fingerprints=[{"fingerprint_id": "shared_fp"}],
        )
        
        # Should find similar failure
        assert len(decision.similar_failures) > 0
        assert decision.similar_failures[0].instance_id == "past_failure"


# =============================================================================
# Self-Critique Rubric Tests
# =============================================================================

class TestSelfCritiqueRubric:
    """Tests for self-critique rubric content."""
    
    def test_rubric_exists(self):
        """Rubric is importable."""
        from rfsn_upstream import SELF_CRITIQUE_RUBRIC
        assert SELF_CRITIQUE_RUBRIC
    
    def test_rubric_content(self):
        """Rubric contains key checks."""
        from rfsn_upstream import SELF_CRITIQUE_RUBRIC
        
        # Must address common issues
        assert "minimal" in SELF_CRITIQUE_RUBRIC.lower()
        assert "exception handling" in SELF_CRITIQUE_RUBRIC.lower()
        assert "root cause" in SELF_CRITIQUE_RUBRIC.lower()
    
    def test_rubric_is_checklist(self):
        """Rubric is formatted as a checklist."""
        from rfsn_upstream import SELF_CRITIQUE_RUBRIC
        
        # Should have numbered items
        assert "1." in SELF_CRITIQUE_RUBRIC or "1)" in SELF_CRITIQUE_RUBRIC


# =============================================================================
# Create Learner State Helper Tests
# =============================================================================

class TestCreateLearnerState:
    """Tests for create_learner_state helper."""
    
    def test_creates_directory_and_dbs(self, tmp_path: Path):
        """Creates directory and initializes DBs."""
        from rfsn_upstream import create_learner_state
        
        new_dir = tmp_path / "new_artifacts"
        bandit_path, outcomes_path = create_learner_state(new_dir)
        
        assert new_dir.exists()
        assert Path(bandit_path).exists()
        assert Path(outcomes_path).exists()
    
    def test_returns_paths(self, tmp_path: Path):
        """Returns correct paths."""
        from rfsn_upstream import create_learner_state
        
        bandit_path, outcomes_path = create_learner_state(tmp_path)
        
        assert bandit_path == str(tmp_path / "bandit.db")
        assert outcomes_path == str(tmp_path / "outcomes.db")

# =============================================================================
# Bandit Factory Tests
# =============================================================================

class TestBanditFactory:
    """Tests for create_bandit factory."""
    
    def test_create_bandit_default_arms(self, tmp_path: Path):
        """Creates bandit with default SWE-bench arms from strategies.py."""
        from rfsn_upstream.bandit import create_bandit
        from rfsn_upstream.strategies import list_arm_ids
        
        bandit = create_bandit(db_path=tmp_path / "bandit.db")
        
        # Should have arms from strategies.py
        expected_arms = set(list_arm_ids())
        bandit_arms = {a.arm_id for a in bandit.get_all_arms()}
        
        # Check subset because list_arm_ids might grow, but we expect at least the standard ones
        assert "swe_minimal_diff" in bandit_arms
        assert "swe_traceback_first" in bandit_arms
        # They should match exactly since create_bandit uses list_arm_ids()
        assert bandit_arms == expected_arms
