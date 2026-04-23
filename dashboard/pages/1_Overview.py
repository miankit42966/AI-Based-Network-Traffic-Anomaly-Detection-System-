from __future__ import annotations

import streamlit as st

from dashboard.common import (
    apply_theme,
    bootstrap_state,
    build_comparison_chart,
    build_comparison_table,
    build_heatmap,
    build_traffic_chart,
    load_frame,
    load_metrics,
    realtime_status,
    render_sidebar,
    require_authentication,
    selected_config,
    show_toasts,
)
from dashboard.db import init_database

st.set_page_config(page_title='Overview', layout='wide')
init_database()
bootstrap_state()
require_authentication()
theme = apply_theme()
render_sidebar('Overview')
show_toasts()

cfg = selected_config()
metrics = load_metrics(cfg['metrics'])
frame = load_frame(cfg['snapshot'])
comparison_df = build_comparison_table()
status_level, status_message = realtime_status(metrics)

st.markdown(
    f"""
    <div class="hero">
      <div class="section-label">Overview</div>
      <h1>{st.session_state['selected_dataset']} Monitoring Overview</h1>
      <p>This page gives you a high-level view of project health, dataset comparison, and the end-to-end phase flow in one place. Responsive cards automatically stack on smaller screens.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if status_level == 'success':
    st.success(status_message)
else:
    st.warning(status_message)

left, right = st.columns([1.0, 2.0], gap='large')
with left:
    st.markdown('### Phase Flow')
    phase_cards = [
        ('Phase 1', 'Data Collection & Dataset', 'CICIDS-2017/2018, UNSW-NB15, KDD Cup99, live capture'),
        ('Phase 2', 'Preprocessing & Feature Engineering', 'Cleaning, scaling, label encoding, flow metrics, SMOTE'),
        ('Phase 3', 'AI Model Training', 'Isolation Forest, LSTM Autoencoder, XGBoost, ensemble fusion'),
        ('Phase 4', 'Real-Time Detection Engine', 'Live ingestion, anomaly scoring, congestion detection, alerts'),
        ('Phase 5', 'Monitoring Dashboard', 'Live charts, alert stream, metrics, heatmap'),
        ('Phase 6', 'Testing & Evaluation', 'Accuracy, F1, ROC-AUC, latency benchmarks'),
    ]
    palette = [
        'linear-gradient(135deg, rgba(37,99,235,0.18), rgba(125,211,252,0.08))',
        'linear-gradient(135deg, rgba(8,145,178,0.18), rgba(125,211,252,0.08))',
        'linear-gradient(135deg, rgba(96,165,250,0.18), rgba(59,130,246,0.08))',
        'linear-gradient(135deg, rgba(56,189,248,0.16), rgba(37,99,235,0.08))',
        'linear-gradient(135deg, rgba(14,165,233,0.16), rgba(103,232,249,0.08))',
        'linear-gradient(135deg, rgba(59,130,246,0.18), rgba(8,145,178,0.08))',
    ]
    borders = ['rgba(125,211,252,0.30)', 'rgba(103,232,249,0.30)', 'rgba(96,165,250,0.28)', 'rgba(125,211,252,0.26)', 'rgba(56,189,248,0.30)', 'rgba(59,130,246,0.28)']
    for idx, (phase, title, subtitle) in enumerate(phase_cards):
        st.markdown(
            f"""
            <div class="phase-card" style="background:{palette[idx]}; border-color:{borders[idx]};">
              <div style="color:{theme['muted']}; font-size:0.85rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">{phase}</div>
              <div class="phase-title">{title}</div>
              <div class="phase-sub">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if idx < len(phase_cards) - 1:
            st.markdown(
                f"<div style='text-align:center; color:{theme['accent_alt']}; font-size:1.5rem; font-weight:800; text-shadow:0 0 16px rgba(96,165,250,0.30);'>&#8595;</div>",
                unsafe_allow_html=True,
            )

with right:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('F1 Score', f"{metrics['f1']:.3f}")
    m2.metric('ROC-AUC', f"{metrics['roc_auc']:.3f}")
    m3.metric('Avg Latency', f"{metrics['latency_ms_avg']:.2f} ms")
    m4.metric('P99 Latency', f"{metrics['latency_ms_p99']:.2f} ms")

    c1, c2 = st.columns([1.35, 1], gap='large')
    with c1:
        st.markdown('### Live Traffic Lens')
        st.plotly_chart(build_traffic_chart(frame, cfg['accent'], st.session_state['selected_dataset'], theme), width='stretch')
    with c2:
        st.markdown('### Evaluation Heatmap')
        st.plotly_chart(build_heatmap(metrics, theme), width='stretch')

    st.markdown('### Dataset Comparison Snapshot')
    z1, z2 = st.columns([1.3, 1], gap='large')
    with z1:
        st.plotly_chart(build_comparison_chart(comparison_df, theme), width='stretch')
    with z2:
        st.dataframe(comparison_df, width='stretch', hide_index=True)
