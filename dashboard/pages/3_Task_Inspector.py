import streamlit as st
import json
import pandas as pd
from pathlib import Path
from dashboard.shared import parse_ledger, list_ledgers, LEDGER_DIR

st.set_page_config(page_title="Task Inspector", page_icon="📜", layout="wide")
st.title("Task Inspector 📜")

# Workspace selector (could be improved to be dynamic, but fixed for now or input)
workspace = st.text_input("Workspace Path", value=".", help="Path to the repository workspace")
workspace_path = Path(workspace)

if not workspace_path.exists():
    st.error(f"Workspace path does not exist: {workspace_path}")
else:
    ledgers = list_ledgers(workspace_path)
    
    if not ledgers:
        st.info("No ledgers found in this workspace (.rfsn_ledger directory).")
    else:
        selected_ledger = st.selectbox(
            "Select Task Ledger",
            ledgers,
            format_func=lambda p: f"{p.name} ({p.stat().st_size} bytes)"
        )
        
        if selected_ledger:
            entries = parse_ledger(selected_ledger)
            
            if not entries:
                st.warning("Ledger file is empty or invalid.")
            else:
                st.success(f"Loaded {len(entries)} events.")
                
                # --- High Level Summary ---
                # Filter for key events
                proposals = [e for e in entries if e.get("type") == "proposal"]
                decisions = [e for e in entries if e.get("type") == "gate_decision"]
                results = [e for e in entries if e.get("type") == "execution_result"]
                
                st.metric("Total Steps", len(decisions))
                st.metric("Accepted Proposals", len([d for d in decisions if d["decision"]["accepted"]]))
                
                # --- Timeline View ---
                st.subheader("Execution Timeline")
                
                for i, entry in enumerate(entries):
                    evt_type = entry.get("type", "unknown")
                    
                    with st.expander(f"Step {i+1}: {evt_type.upper()}", expanded=(i==len(entries)-1)):
                        # Custom formatting based on type
                        if evt_type == "proposal":
                            prop = entry.get("proposal", {})
                            st.markdown(f"**Intent:** `{prop.get('intent')}`")
                            st.markdown(f"**Target:** `{prop.get('target')}`")
                            st.caption("-- Reason --")
                            st.write(prop.get("justification"))
                            if prop.get("patch"):
                                st.code(prop.get("patch"), language="diff")
                                
                        elif evt_type == "gate_decision":
                            decision = entry.get("decision", {})
                            accepted = decision.get("accepted")
                            color = "green" if accepted else "red"
                            st.markdown(f":{color}[{'ACCEPTED' if accepted else 'REJECTED'}]")
                            st.markdown(f"**Reason:** {decision.get('reason')}")
                            
                        elif evt_type == "execution_result":
                            res = entry.get("result", {})
                            st.markdown(f"**Status:** `{res.get('status')}`")
                            st.code(res.get("stdout") or res.get("stderr"), language="text")
                            
                        else:
                            st.json(entry)

