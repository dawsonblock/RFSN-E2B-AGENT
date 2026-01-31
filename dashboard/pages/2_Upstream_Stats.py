import streamlit as st
import pandas as pd
import altair as alt
from dashboard.shared import load_bandit_stats, load_recent_history, load_outcomes

st.set_page_config(page_title="Upstream Stats", page_icon="📊", layout="wide")
st.title("Upstream Learning Statistics 📊")

# Load data
arms_df = load_bandit_stats()
history_df = load_recent_history()
outcomes_df = load_outcomes()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Strategy Performance (Bandit)")
    if not arms_df.empty:
        # Sort by success rate
        arms_df = arms_df.sort_values(by="win_rate", ascending=False)
        
        # Display as table
        st.dataframe(
            arms_df[[
                "arm_id", "win_rate", "pulls", "successes", "failures", "alpha", "beta"
            ]].style.format({
                "win_rate": "{:.2%}",
                "alpha": "{:.1f}",
                "beta": "{:.1f}"
            }),
            use_container_width=True
        )
        
        # Chart
        chart = alt.Chart(arms_df).mark_bar().encode(
            x=alt.X('arm_id', sort='-y'),
            y=alt.Y('win_rate', axis=alt.Axis(format='%')),
            tooltip=['arm_id', 'win_rate', 'pulls']
        ).properties(title="Success Rate by Strategy")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No bandit data found. Have you run any episodes?")

with col2:
    st.subheader("Recent Activity")
    if not history_df.empty:
        st.dataframe(
            history_df[["timestamp", "task_id", "arm_id", "success", "reward"]],
            use_container_width=True
        )
    else:
        st.info("No recent history.")

st.divider()

st.subheader("Outcome Analysis")
if not outcomes_df.empty:
    # Time series of success
    outcomes_df['timestamp'] = pd.to_datetime(outcomes_df['timestamp'])
    
    line_chart = alt.Chart(outcomes_df).mark_line(point=True).encode(
        x='timestamp',
        y='success:Q',
        tooltip=['task_id', 'arm_id', 'success']
    ).properties(title="Success Over Time")
    
    st.altair_chart(line_chart, use_container_width=True)
    
    # Fingerprint analysis
    st.subheader("Failure Fingerprints")
    # This would require parsing the nested JSON in fingerprints column if we wanted detail,
    # but for now let's just show the raw table
    st.dataframe(outcomes_df, use_container_width=True)

else:
    st.info("No outcome records found.")
