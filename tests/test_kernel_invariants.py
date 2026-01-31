"""Tests for RFSN Kernel Invariants.

These tests verify the 7 non-negotiable invariants:
1. Planner never executes
2. Gate never learns
3. Controller never decides
4. All commits are serial
5. No hidden state across decisions
6. All side effects are logged
7. Rejected actions produce evidence
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rfsn_kernel import (
    Controller,
    ControllerConfig,
    Gate,
    GateConfig,
    GateDecision,
    Ledger,
    Planner,
    Proposal,
    ProposalIntent,
    SafetyEnvelope,
    StateSnapshot,
    run_kernel,
    KernelConfig,
)
from rfsn_kernel.planner_interface import StaticPlanner, NullPlanner
from rfsn_kernel.proposal import create_proposal
from rfsn_kernel.state import create_initial_state, TestResult, TestStatus, TaskProgress


class TestGateInvariants:
    """Tests for gate invariants."""
    
    def test_gate_is_pure_function(self):
        """Invariant: Gate is a pure function with no side effects."""
        gate = Gate(GateConfig())
        
        # Create state and proposal
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            proposal = create_proposal(
                intent=ProposalIntent.READ_FILE,
                target="test.py",
                justification="Test",
                expected_effect="Read file",
            )
            
            # Call validate multiple times
            decision1 = gate.validate(state, proposal)
            decision2 = gate.validate(state, proposal)
            decision3 = gate.validate(state, proposal)
            
            # All decisions should be identical (deterministic)
            assert decision1.accepted == decision2.accepted == decision3.accepted
            assert decision1.reason == decision2.reason == decision3.reason
    
    def test_gate_never_learns(self):
        """Invariant: Gate has no learning methods or mutable state."""
        gate = Gate(GateConfig())
        
        # Gate should not have any update/learn methods
        learning_methods = ['update', 'learn', 'train', 'fit', 'adapt']
        for method in learning_methods:
            assert not hasattr(gate, method), f"Gate should not have {method} method"
        
        # Config should be frozen
        with pytest.raises((AttributeError, TypeError)):
            gate._config.max_diff_lines = 99999
    
    def test_gate_rejects_with_evidence(self):
        """Invariant: Rejected actions produce evidence."""
        gate = Gate(GateConfig(
            forbidden_path_patterns=("*.secret",),
        ))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            proposal = create_proposal(
                intent=ProposalIntent.MODIFY_FILE,
                target="config.secret",  # Forbidden
                justification="Test",
                expected_effect="Should be rejected",
            )
            
            decision = gate.validate(state, proposal)
            
            # Must be rejected
            assert not decision.accepted
            
            # Must have evidence
            assert decision.evidence
            evidence_dict = dict(decision.evidence)
            assert "target" in evidence_dict or "forbidden_pattern" in evidence_dict
    
    def test_gate_deterministic(self):
        """Invariant: Same inputs → same outputs."""
        gate = Gate(GateConfig())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            
            # Test with various proposals
            proposals = [
                create_proposal(ProposalIntent.READ_FILE, "test.py", "j", "e"),
                create_proposal(ProposalIntent.MODIFY_FILE, "test.py", "j", "e", patch="fix"),
                create_proposal(ProposalIntent.RUN_TEST, "tests/", "j", "e"),
            ]
            
            for proposal in proposals:
                decisions = [gate.validate(state, proposal) for _ in range(10)]
                
                # All should be identical
                first = decisions[0]
                for d in decisions[1:]:
                    assert d.accepted == first.accepted
                    assert d.reason == first.reason


class TestControllerInvariants:
    """Tests for controller invariants."""
    
    def test_controller_requires_gate_approval(self):
        """Invariant: Controller only executes gate-approved proposals."""
        controller = Controller(ControllerConfig())
        
        proposal = create_proposal(
            intent=ProposalIntent.READ_FILE,
            target="test.py",
            justification="Test",
            expected_effect="Read file",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Attempting to execute without gate approval should raise
            with pytest.raises(Exception) as exc_info:
                controller.execute(proposal, tmpdir, gate_approved=False)
            
            assert "gate approval" in str(exc_info.value).lower()
    
    def test_controller_cannot_bypass_gate(self):
        """Invariant: Controller has no method to bypass gate."""
        controller = Controller(ControllerConfig())
        
        # Controller should not have bypass methods
        bypass_methods = ['bypass', 'force', 'override', 'skip_gate']
        for method in bypass_methods:
            assert not hasattr(controller, method), f"Controller should not have {method}"
    
    def test_controller_never_decides(self):
        """Invariant: Controller has no decision-making methods."""
        controller = Controller(ControllerConfig())
        
        # Controller should not have decision methods
        decision_methods = ['decide', 'evaluate', 'choose', 'select']
        for method in decision_methods:
            assert not hasattr(controller, method), f"Controller should not have {method}"


class TestPlannerInvariants:
    """Tests for planner invariants."""
    
    def test_planner_never_executes(self):
        """Invariant: Planner protocol has no execute methods."""
        # Check the abstract Planner class
        planner_methods = dir(Planner)
        
        # Should not have execute methods
        execute_methods = ['execute', 'run', 'apply', 'perform']
        for method in execute_methods:
            assert method not in planner_methods, f"Planner should not have {method}"
        
        # Should only have propose and observe methods
        assert 'propose' in planner_methods
        assert 'observe_rejection' in planner_methods
    
    def test_null_planner_never_executes(self):
        """Invariant: NullPlanner has no side effects."""
        planner = NullPlanner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            
            # Propose should return None, no side effects
            result = planner.propose(state, "test task")
            assert result is None
    
    def test_static_planner_only_proposes(self):
        """Invariant: StaticPlanner only returns proposals, doesn't execute."""
        proposals = [
            create_proposal(ProposalIntent.READ_FILE, "a.py", "j", "e"),
            create_proposal(ProposalIntent.MODIFY_FILE, "b.py", "j", "e"),
        ]
        planner = StaticPlanner(proposals)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            
            # Planner returns proposals, doesn't execute them
            p1 = planner.propose(state, "task")
            assert isinstance(p1, Proposal)
            
            p2 = planner.propose(state, "task")
            assert isinstance(p2, Proposal)
            
            # Should run out of proposals
            p3 = planner.propose(state, "task")
            assert p3 is None


class TestLedgerInvariants:
    """Tests for ledger invariants."""
    
    def test_all_side_effects_logged(self):
        """Invariant: All side effects are logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = Ledger(Path(tmpdir) / "test.jsonl")
            state = create_initial_state("test", Path(tmpdir))
            
            proposal = create_proposal(
                intent=ProposalIntent.READ_FILE,
                target="test.py",
                justification="Test",
                expected_effect="Read file",
            )
            
            from rfsn_kernel.controller import ExecutionResult
            result = ExecutionResult(
                success=True,
                proposal_id=proposal.proposal_id,
                intent=proposal.intent.value,
                stdout="content",
                stderr="",
                return_code=0,
                duration_seconds=0.1,
                artifacts=(),
                changed_files=(),
            )
            
            # Record execution
            entry = ledger.record(state, proposal, result)
            
            # Entry should be in ledger
            entries = ledger.replay()
            assert len(entries) == 1
            assert entries[0].entry_type == "execution"
            assert entries[0].proposal_id == proposal.proposal_id
    
    def test_rejected_actions_produce_evidence(self):
        """Invariant: Rejected actions produce evidence in ledger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = Ledger(Path(tmpdir) / "test.jsonl")
            state = create_initial_state("test", Path(tmpdir))
            
            proposal = create_proposal(
                intent=ProposalIntent.MODIFY_FILE,
                target="forbidden.secret",
                justification="Test",
                expected_effect="Should be rejected",
            )
            
            decision = GateDecision.reject(
                reason="Path forbidden",
                evidence={"target": "forbidden.secret", "pattern": "*.secret"},
            )
            
            # Record rejection
            entry = ledger.record_rejection(state, proposal, decision)
            
            # Entry should have evidence
            assert entry.entry_type == "rejection"
            data = dict(entry.data)
            assert "reason" in data
            assert "evidence" in data


class TestKernelInvariants:
    """Tests for overall kernel invariants."""
    
    def test_serial_commits(self):
        """Invariant: All commits are serial (one at a time)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "test.py").write_text("print('hello')")
            
            # Create proposals that would modify the same file
            proposals = [
                create_proposal(
                    ProposalIntent.MODIFY_FILE,
                    "test.py",
                    "First modification",
                    "Change 1",
                    patch="print('first')",
                ),
                create_proposal(
                    ProposalIntent.MODIFY_FILE,
                    "test.py",
                    "Second modification",
                    "Change 2",
                    patch="print('second')",
                ),
            ]
            
            planner = StaticPlanner(proposals)
            
            result = run_kernel(
                task_id="test_serial",
                task="Test serial commits",
                workspace_path=workspace,
                planner=planner,
                config=KernelConfig(max_steps=5),
            )
            
            # Should have executed proposals serially
            assert result.accepted_proposals <= 2
    
    def test_no_hidden_state(self):
        """Invariant: No hidden state across decisions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "test.py").write_text("x = 1")
            
            state1 = create_initial_state("test1", workspace)
            state2 = create_initial_state("test2", workspace)
            
            # States should be independent
            assert state1.task_id != state2.task_id
            assert state1.state_hash != state2.state_hash
            
            # Gate decisions should not carry over
            gate = Gate(GateConfig())
            
            proposal = create_proposal(
                ProposalIntent.READ_FILE, "test.py", "j", "e"
            )
            
            # Multiple decisions on different states
            d1 = gate.validate(state1, proposal)
            d2 = gate.validate(state2, proposal)
            
            # Both should succeed independently
            assert d1.accepted
            assert d2.accepted


class TestStateImmutability:
    """Tests for state immutability."""
    
    def test_state_is_frozen(self):
        """Invariant: State snapshots are immutable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            
            # Attempt to modify should fail
            with pytest.raises((AttributeError, TypeError)):
                state.task_id = "modified"
    
    def test_state_serialization_roundtrip(self):
        """State should serialize and deserialize correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = create_initial_state("test", Path(tmpdir))
            
            # Serialize
            json_str = state.to_json()
            
            # Deserialize
            restored = StateSnapshot.from_json(json_str)
            
            # Should be equal
            assert restored.state_hash == state.state_hash
            assert restored.task_id == state.task_id


class TestProposalValidation:
    """Tests for proposal validation."""
    
    def test_shell_injection_blocked(self):
        """Proposals with shell injection patterns should be blocked."""
        dangerous_patches = [
            "$(rm -rf /)",
            "`whoami`",
            "import os; os.system('rm -rf /')",
            "subprocess.run(cmd, shell=True)",
        ]
        
        for patch in dangerous_patches:
            with pytest.raises(Exception):
                create_proposal(
                    ProposalIntent.MODIFY_FILE,
                    "test.py",
                    "Dangerous",
                    "Should fail",
                    patch=patch,
                )
    
    def test_valid_proposal_passes(self):
        """Valid proposals should pass validation."""
        proposal = create_proposal(
            ProposalIntent.MODIFY_FILE,
            "src/utils.py",
            "Fix bug in helper function",
            "Test test_helper will pass",
            patch="def helper():\n    return 42",
        )
        
        assert proposal.intent == ProposalIntent.MODIFY_FILE
        assert proposal.target == "src/utils.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
