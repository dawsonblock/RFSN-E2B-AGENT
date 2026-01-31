"""Tests for Upstream Learner components."""

import tempfile
from pathlib import Path

import pytest

from rfsn_upstream import (
    ThompsonBandit,
    ArmStats,
    Fingerprint,
    compute_fingerprint,
    fingerprint_from_rejection,
    Memory,
    MemoryIndex,
    PROMPT_VARIANTS,
    get_variant,
    PromptVariant,
)
from rfsn_upstream.retrieval import create_memory


class TestThompsonBandit:
    """Tests for Thompson Sampling bandit."""
    
    def test_bandit_initialization(self):
        """Bandit should initialize with arms."""
        arms = ["arm1", "arm2", "arm3"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bandit = ThompsonBandit(
                db_path=Path(tmpdir) / "test.db",
                arms=arms,
            )
            
            all_arms = bandit.get_all_arms()
            assert len(all_arms) == 3
            
            for arm in all_arms:
                assert arm.arm_id in arms
                assert arm.alpha == 1.0  # Prior
                assert arm.beta == 1.0   # Prior
    
    def test_bandit_selection(self):
        """Bandit should select arms via Thompson Sampling."""
        arms = ["arm1", "arm2"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bandit = ThompsonBandit(Path(tmpdir) / "test.db", arms)
            
            # Select arms multiple times
            selections = [bandit.select_arm() for _ in range(100)]
            
            # Both arms should be selected at least once
            assert "arm1" in selections
            assert "arm2" in selections
    
    def test_bandit_update(self):
        """Bandit should update arm stats correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bandit = ThompsonBandit(Path(tmpdir) / "test.db", ["arm1"])
            
            # Initial state
            arm = bandit.get_arm_stats("arm1")
            assert arm.alpha == 1.0
            assert arm.beta == 1.0
            assert arm.pulls == 0
            
            # Update with success
            bandit.update("arm1", success=True)
            arm = bandit.get_arm_stats("arm1")
            assert arm.alpha == 2.0  # Increased
            assert arm.beta == 1.0   # Unchanged
            assert arm.pulls == 1
            assert arm.successes == 1
            
            # Update with failure
            bandit.update("arm1", success=False)
            arm = bandit.get_arm_stats("arm1")
            assert arm.alpha == 2.0  # Unchanged
            assert arm.beta == 2.0   # Increased
            assert arm.pulls == 2
            assert arm.failures == 1
    
    def test_bandit_persistence(self):
        """Bandit state should persist across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Create and update
            bandit1 = ThompsonBandit(db_path, ["arm1"])
            bandit1.update("arm1", success=True)
            bandit1.update("arm1", success=True)
            
            # Create new instance
            bandit2 = ThompsonBandit(db_path)
            arm = bandit2.get_arm_stats("arm1")
            
            assert arm.alpha == 3.0  # 1 prior + 2 successes
            assert arm.successes == 2
    
    def test_bandit_export_import(self):
        """Bandit state should export and import correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bandit = ThompsonBandit(Path(tmpdir) / "test.db", ["a", "b"])
            bandit.update("a", success=True)
            bandit.update("b", success=False)
            
            # Export
            state = bandit.export_state()
            assert len(state["arms"]) == 2
            
            # Import to new bandit
            new_bandit = ThompsonBandit(Path(tmpdir) / "test2.db")
            new_bandit.import_state(state)
            
            arm_a = new_bandit.get_arm_stats("a")
            assert arm_a.successes == 1


class TestFingerprinting:
    """Tests for failure fingerprinting."""
    
    def test_fingerprint_deterministic(self):
        """Fingerprints should be deterministic."""
        fp1 = compute_fingerprint("test_failure", "NameError: name 'x' is not defined")
        fp2 = compute_fingerprint("test_failure", "NameError: name 'x' is not defined")
        
        assert fp1.fingerprint_id == fp2.fingerprint_id
        assert fp1.category == fp2.category
    
    def test_fingerprint_classification(self):
        """Fingerprints should classify errors correctly."""
        # Python errors
        fp = compute_fingerprint("test_failure", "SyntaxError: invalid syntax")
        assert fp.category == "syntax_error"
        
        fp = compute_fingerprint("test_failure", "NameError: name 'foo' is not defined")
        assert fp.category == "name_error"
        
        # Gate rejections
        fp = compute_fingerprint("gate_rejection", "Path '/etc/passwd' matches forbidden pattern")
        assert fp.category == "path_violation"
    
    def test_fingerprint_pattern_extraction(self):
        """Fingerprints should extract patterns."""
        fp = compute_fingerprint(
            "test_failure",
            "File 'utils.py', line 42, in process\n  NameError: name 'result' is not defined"
        )
        
        patterns = fp.patterns
        assert any("file:utils.py" in p for p in patterns)
        assert any("line:42" in p for p in patterns)
        assert any("name:result" in p for p in patterns)
    
    def test_fingerprint_similarity(self):
        """Similar fingerprints should have high similarity."""
        fp1 = compute_fingerprint("test_failure", "NameError: name 'x' is not defined")
        fp2 = compute_fingerprint("test_failure", "NameError: name 'y' is not defined")
        fp3 = compute_fingerprint("test_failure", "SyntaxError: invalid syntax")
        
        # Same category should be similar
        sim12 = fp1.similarity(fp2)
        sim13 = fp1.similarity(fp3)
        
        assert sim12 > sim13  # Same error type more similar


class TestMemoryRetrieval:
    """Tests for memory storage and retrieval."""
    
    def test_memory_storage(self):
        """Memories should be stored and retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = MemoryIndex(Path(tmpdir) / "memories.db")
            
            memory = create_memory(
                memory_type="success",
                task_id="task_001",
                content="Fixed bug by changing return value",
            )
            
            index.store(memory)
            
            retrieved = index.get(memory.memory_id)
            assert retrieved is not None
            assert retrieved.content == memory.content
    
    def test_memory_search(self):
        """Memories should be searchable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = MemoryIndex(Path(tmpdir) / "memories.db")
            
            index.store(create_memory("success", "t1", "Fixed NameError in parser"))
            index.store(create_memory("failure", "t2", "Failed due to timeout"))
            index.store(create_memory("success", "t3", "Fixed TypeError in handler"))
            
            # Search for specific term
            results = index.search("parser")
            assert len(results) >= 1
            assert any("parser" in r.content.lower() for r in results)
    
    def test_memory_by_type(self):
        """Memories should be filterable by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = MemoryIndex(Path(tmpdir) / "memories.db")
            
            index.store(create_memory("success", "t1", "Success 1"))
            index.store(create_memory("success", "t2", "Success 2"))
            index.store(create_memory("failure", "t3", "Failure 1"))
            
            successes = index.get_recent(k=10, memory_type="success")
            assert len(successes) == 2
            assert all(m.memory_type == "success" for m in successes)


class TestPromptVariants:
    """Tests for prompt variants."""
    
    def test_all_variants_exist(self):
        """All expected variants should exist."""
        expected = [
            "v_minimal_fix",
            "v_diagnose_then_patch",
            "v_test_first",
            "v_multi_hypothesis",
            "v_repair_loop",
        ]
        
        for name in expected:
            variant = get_variant(name)
            assert isinstance(variant, PromptVariant)
            assert variant.name == name
    
    def test_variant_has_prompts(self):
        """Variants should have system and user prompts."""
        for name, variant in PROMPT_VARIANTS.items():
            assert variant.system_prompt, f"{name} missing system_prompt"
            assert variant.user_prompt_template, f"{name} missing user_prompt_template"
    
    def test_variant_formatting(self):
        """Prompt variants should format correctly."""
        from rfsn_upstream.prompt_variants import format_prompt
        
        variant = get_variant("v_minimal_fix")
        
        system, user = format_prompt(
            variant,
            problem_statement="Fix the bug",
            file_content="def foo(): pass",
        )
        
        assert "Fix the bug" in user
        assert "def foo()" in user


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
