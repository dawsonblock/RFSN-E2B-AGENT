import streamlit as st
import subprocess
import shlex
import os
from pathlib import Path

st.set_page_config(page_title="Agent Launcher", page_icon="🚀", layout="wide")
st.title("Agent Launcher 🚀")

with st.form("launcher_form"):
    st.markdown("### Task Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        task_id = st.text_input("Task ID", value="TASK_001")
        workspace = st.text_input("Workspace Path", value=".")
    
    with col2:
        provider = st.selectbox("LLM Provider", ["deepseek", "openai", "anthropic"])
        arm_id = st.selectbox(
            "Strategy Arm", 
            ["(Auto-select)", "swe_minimal_diff", "swe_traceback_first", "swe_contract_fix", "swe_regression_guard", "swe_test_driven", "swe_import_fix", "swe_type_fix"]
        )
    
    task_desc = st.text_area("Task Description", value="Fix the failing test in tests/test_foo.py", height=100)
    
    submitted = st.form_submit_button("Launch Agent")

if submitted:
    # 1. Construct command
    cmd = [
        "python", "run_agent.py",
        "--task-id", task_id,
        "--workspace", workspace,
        "--task", task_desc,
        "--provider", provider
    ]
    
    if arm_id != "(Auto-select)":
        cmd.extend(["--arm-id", arm_id])
        
    cmd_str = shlex.join(cmd)
    
    st.info(f"Running command: `{cmd_str}`")
    
    # 2. Run and stream output
    output_container = st.empty()
    logs = []
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd()  # Run from repo root
        )
        
        # Stream output
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logs.append(line)
                # Keep only last 20 lines in preview to avoid UI lag, but user can see full logs below
                output_container.code("".join(logs[-20:]), language="text")
                
        return_code = process.poll()
        
        if return_code == 0:
            st.success("Agent completed successfully! 🎉")
        else:
            st.error(f"Agent failed with return code {return_code}")
            
        with st.expander("Full Execution Log"):
            st.code("".join(logs))
            
    except Exception as e:
        st.error(f"Failed to launch: {e}")
