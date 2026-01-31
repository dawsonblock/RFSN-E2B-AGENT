<div align="center">

# 🛡️ RFSN E2B Agent

### Deterministic Agent Kernel for Autonomous Code Repair

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![E2B Ready](https://img.shields.io/badge/E2B-Ready-ff6b35.svg?style=for-the-badge)](https://e2b.dev)

<p align="center">
  <strong>A safety-first kernel that separates thinking from doing</strong><br>
  <sub>Built for SWE-bench • 7 Non-Negotiable Invariants • Zero Trust Architecture</sub>
</p>

</div>

---

## 🎯 What is RFSN?

**RFSN** (Robust Fail-Safe Network) is a deterministic agent kernel that enables autonomous code repair while enforcing strict safety guarantees. The architecture separates:

- 🧠 **Planning** (LLM proposes changes)
- 🚦 **Gating** (Pure function validates safety)
- ⚡ **Execution** (Sandboxed controller applies changes)
- 📚 **Learning** (Upstream learner optimizes prompts)

```
┌─────────────────────────────────────────────────────────┐
│              Upstream Learner (Learning)                │
│   🎰 Bandit  ─→  📋 Prompts  ─→  🧠 LLM Planner        │
└──────────────────────┬──────────────────────────────────┘
                       │ proposal
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 RFSN Kernel (Safety)                    │
│   🚦 Gate  ─→  ⚡ Controller  ─→  📝 Ledger            │
│   (pure)       (sandboxed)        (audit)              │
└─────────────────────────────────────────────────────────┘
```

---

## 🔒 7 Non-Negotiable Invariants

| # | Invariant | Enforcement |
|:-:|-----------|-------------|
| 1 | **Planner never executes** | No execute methods in Planner class |
| 2 | **Gate never learns** | Frozen config, no update methods |
| 3 | **Controller never decides** | Requires `gate_approved=True` |
| 4 | **All commits are serial** | Single proposal per iteration |
| 5 | **No hidden state** | Immutable `StateSnapshot` with hash |
| 6 | **All side effects logged** | JSONL ledger with replay |
| 7 | **Rejections produce evidence** | Evidence dict on every reject |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/dawsonblock/RFSN-E2B-AGENT.git
cd RFSN-E2B-AGENT

# Install
pip install -e .

# Set API key
export DEEPSEEK_API_KEY="your-key"

# Run
python run_agent.py \
    --task-id BUG_001 \
    --workspace ./your_repo \
    --task "Fix the failing test in utils.py"
```

---

## 📦 Package Structure

```
rfsn_kernel/           # 🛡️ Core safety kernel (non-learning)
├── state.py           # Immutable state with SHA256 hash
├── proposal.py        # Schema + shell injection detection
├── gate.py            # Pure deterministic safety gate
├── controller.py      # Sandboxed execution
├── ledger.py          # Ground truth JSONL logging
├── kernel.py          # Main orchestration loop
└── llm_planner.py     # DeepSeek/OpenAI/Anthropic

rfsn_upstream/         # 🎰 Learning layer (separate)
├── bandit.py          # Thompson Sampling + SQLite
├── fingerprint.py     # Failure classification
├── retrieval.py       # Memory with FTS search
└── prompt_variants.py # 7 SWE-bench prompts
```

---

## 🧪 SWE-bench Integration

```bash
# Run with bandit learning
python run_agent.py \
    --task-id django__django-12345 \
    --workspace ./django \
    --task "Fix the bug" \
    --bandit-db ./bandit.db

# Or use episode runner for CI
python scripts/run_swebench_episode.py \
    --instance-id django__django-12345 \
    --repo ./django \
    --arm-id v_minimal_fix \
    --output outcome.json
```

### Prompt Variants

| Variant | Strategy |
|---------|----------|
| `v_minimal_fix` | Single-line minimal changes |
| `v_diagnose_then_patch` | Root cause analysis first |
| `v_test_first` | Understand test expectations |
| `v_multi_hypothesis` | Generate 3 hypotheses, pick best |
| `v_repair_loop` | Learn from rejection feedback |
| `v_context_aware` | Use retrieved similar bugs |
| `v_chain_of_thought` | Explicit reasoning chain |

---

## ⚙️ Configuration

### Environment Variables

```bash
DEEPSEEK_API_KEY=...    # DeepSeek (recommended)
OPENAI_API_KEY=...      # OpenAI (optional)
ANTHROPIC_API_KEY=...   # Anthropic (optional)
```

### CLI Options

```bash
python run_agent.py \
    --task-id ID          # Unique task identifier
    --workspace PATH      # Repository path
    --task "description"  # What to fix
    --provider deepseek   # LLM provider
    --model MODEL         # Model name
    --max-steps 20        # Max iterations
    --max-patches 10      # Max file changes
    --bandit-db PATH      # Enable learning
    --verbose             # Debug logging
```

---

## 🔬 Verification

```bash
# Run tests
python -m pytest tests/ -v

# Quick verification
python -c "
from rfsn_kernel import Gate, GateConfig
gate = Gate(GateConfig())
print('✅ Gate initialized')
print('   Learning methods:', [m for m in dir(gate) if 'learn' in m])  # []
"
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with 🔒 safety-first architecture</sub>
</div>
