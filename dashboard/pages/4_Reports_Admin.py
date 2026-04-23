from __future__ import annotations

import streamlit as st

from dashboard.common import (
    apply_theme,
    bootstrap_state,
    build_report_downloads,
    fetch_history_df,
    load_metrics,
    parse_classification_report,
    render_sidebar,
    require_authentication,
    selected_config,
    show_toasts,
)
from dashboard.db import fetch_users, init_database

st.set_page_config(page_title='Reports & Admin', layout='wide')
init_database()
bootstrap_state()
require_authentication()
apply_theme()
render_sidebar('Reports & Admin')
show_toasts()

cfg = selected_config()
metrics = load_metrics(cfg['metrics'])
class_df = parse_classification_report(metrics.get('classification_report', ''))
csv_bytes, pdf_bytes, report_df = build_report_downloads(st.session_state['selected_dataset'], metrics, class_df)
history_df = fetch_history_df(limit=25)
users = fetch_users()

st.markdown(
    f"""
    <div class="hero">
      <div class="section-label">Reports and Administration</div>
      <h1>{st.session_state['selected_dataset']} Report Center</h1>
      <p>This page centralizes CSV/PDF exports, admin user information, and SQLite-backed recent training logs. It is especially useful for presentations and project review demos.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

a1, a2 = st.columns([1, 1], gap='large')
with a1:
    st.markdown('### Export Reports')
    st.download_button('Download CSV Report', data=csv_bytes, file_name=f"{st.session_state['selected_dataset'].lower().replace(' ', '_')}_report.csv", mime='text/csv', use_container_width=True)
    st.download_button('Download PDF Report', data=pdf_bytes, file_name=f"{st.session_state['selected_dataset'].lower().replace(' ', '_')}_report.pdf", mime='application/pdf', use_container_width=True)
with a2:
    st.markdown('### Admin Users')
    st.dataframe(users, width='stretch', hide_index=True)
    st.caption('The demo login is seeded locally in SQLite for project presentation flow.')

b1, b2 = st.columns([1.2, 1], gap='large')
with b1:
    st.markdown('### Report Preview')
    st.dataframe(report_df, width='stretch', hide_index=True)
with b2:
    st.markdown('### Recent Training Logs')
    if history_df.empty:
        st.info('No training history recorded yet.')
    else:
        preview = history_df.sort_values('timestamp', ascending=False).copy()
        preview['timestamp'] = preview['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(preview[['timestamp', 'dataset', 'max_rows', 'status', 'f1', 'roc_auc', 'notes']], width='stretch', hide_index=True)
