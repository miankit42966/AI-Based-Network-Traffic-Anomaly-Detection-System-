from __future__ import annotations

import streamlit as st

from dashboard.common import (
    apply_theme,
    bootstrap_state,
    load_label_distribution,
    queue_toast,
    render_sidebar,
    require_authentication,
    selected_config,
    show_toasts,
    trigger_training,
)
from dashboard.db import init_database

st.set_page_config(page_title='Datasets & Training', layout='wide')
init_database()
bootstrap_state()
require_authentication()
apply_theme()
render_sidebar('Datasets & Training')
show_toasts()

cfg = selected_config()
labels = load_label_distribution(cfg['raw_snapshot'])

st.markdown(
    f"""
    <div class="hero">
      <div class="section-label">Training Control</div>
      <h1>{st.session_state['selected_dataset']} Dataset and Retraining</h1>
      <p>Use this page to trigger dataset-specific training, control the row budget, and inspect the raw label distribution. It is designed as a practical control-room workflow for your major project demo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1.05, 1], gap='large')
with c1:
    st.markdown('### Training Orchestrator')
    st.write(f"Selected dataset: `{st.session_state['selected_dataset']}`")
    st.write(f"Training rows: `{st.session_state['max_rows']:,}`")
    if st.button('Run training now', use_container_width=True):
        with st.spinner('Executing the full pipeline...'):
            ok, message = trigger_training(st.session_state['selected_dataset'], st.session_state['max_rows'])
            st.session_state['run_success'] = ok
            st.session_state['run_message'] = message
            queue_toast(message)
        st.rerun()
    if st.session_state.get('run_message'):
        if st.session_state.get('run_success'):
            st.success(st.session_state['run_message'])
        else:
            st.error(st.session_state['run_message'])

with c2:
    st.markdown('### Dataset Notes')
    st.markdown('- `CICIDS-2017`: binary-heavy benchmark shard with very strong current results')
    st.markdown('- `CICIDS-2018`: more challenging multi-class behavior, useful for advanced analysis')
    st.markdown('- `Latest Pipeline Run`: whichever run executed most recently')

st.markdown('### Label Distribution')
st.dataframe(labels, width='stretch', hide_index=True)
