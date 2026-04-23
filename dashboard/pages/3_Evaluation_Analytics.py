from __future__ import annotations

import streamlit as st

from dashboard.common import (
    apply_theme,
    bootstrap_state,
    build_attack_breakdown_chart,
    build_class_metrics_chart,
    build_heatmap,
    build_history_chart,
    fetch_history_df,
    load_label_distribution,
    load_label_names,
    load_metrics,
    parse_classification_report,
    render_sidebar,
    require_authentication,
    selected_config,
    show_toasts,
)
from dashboard.db import init_database

st.set_page_config(page_title='Evaluation Analytics', layout='wide')
init_database()
bootstrap_state()
require_authentication()
theme = apply_theme()
render_sidebar('Evaluation Analytics')
show_toasts()

cfg = selected_config()
metrics = load_metrics(cfg['metrics'])
label_names = load_label_names(cfg['raw_snapshot'])
class_df = parse_classification_report(metrics.get('classification_report', ''), label_names=label_names)
label_df = load_label_distribution(cfg['raw_snapshot'])
history_df = fetch_history_df(dataset_name=st.session_state['selected_dataset'])

st.markdown(
    f"""
    <div class="hero">
      <div class="section-label">Evaluation Analytics</div>
      <h1>{st.session_state['selected_dataset']} Deep Evaluation</h1>
      <p>This page presents class-wise precision, recall, and F1 scores, confusion heatmaps, attack-label distribution, and historical run trends in a dedicated analytics view.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if history_df.empty:
    st.info('No database-backed history is available yet for the selected dataset. Run training from the training page to populate this section.')

total_rows = int(label_df['Count'].sum()) if not label_df.empty and 'Count' in label_df.columns else 0
total_test_rows = int(class_df['Support'].sum()) if not class_df.empty and 'Support' in class_df.columns else 0
if total_rows and total_test_rows:
    st.caption(f'Run interpretation: the attack breakdown shows {total_rows:,} selected rows, while the classification table and confusion heatmap summarize the test split of {total_test_rows:,} rows. A perfect score on a single split means the selected shard is highly separable, not necessarily that the model is universally perfect.')

r1, r2 = st.columns([1.2, 1], gap='large')
with r1:
    st.markdown('### Class-wise Metrics')
    st.plotly_chart(build_class_metrics_chart(class_df, theme), width='stretch')
with r2:
    st.markdown('### Confusion Heatmap')
    st.plotly_chart(build_heatmap(metrics, theme, label_names=label_names), width='stretch')

r3, r4 = st.columns([1.15, 1], gap='large')
with r3:
    st.markdown('### Attack Label Breakdown')
    st.plotly_chart(build_attack_breakdown_chart(label_df, cfg['accent'], theme), width='stretch')
with r4:
    st.markdown('### Classification Table')
    if class_df.empty:
        st.code(metrics.get('classification_report', 'No report available.'))
    else:
        st.dataframe(class_df, width='stretch', hide_index=True)

st.markdown('### Historical Trendlines')
st.plotly_chart(build_history_chart(history_df, theme), width='stretch')
