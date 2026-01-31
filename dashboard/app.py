import streamlit as st
import sys
from pathlib import Path

# Add project root to path so we can import modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="RFSN Operational Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.title("RFSN Operational Dashboard 🧠")

st.markdown("""
Welcome to the RFSN (Retrieval-First Serial Network) operational center.

**Navigation:**

- **🚀 Launcher**: Configure and dispatch new agent tasks manually.
- **📊 Upstream Stats**: Monitor bandit learning, strategy performance, and outcomes.
- **📜 Task Inspector**: Deep dive into execution logs (ledgers) for debugging.

---

### System Status

- **Kernel Mode**: `Strict` (Invariant checks enabled)
- **Upstream Learning**: `Active` (Thompson Sampling enabled)
- **Gate**: `Locked` (Deterministic validation)

""")

st.info("Select a page from the sidebar to get started.")
