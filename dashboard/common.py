from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.db import fetch_run_history, fetch_user, record_run

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / 'data' / 'processed'
REAL_RUNS_DIR = PROCESSED_DIR / 'real_runs'
MODELS_DIR = PROCESSED_DIR / 'models'

DATASET_OPTIONS = {
    'Latest Pipeline Run': {
        'metrics': MODELS_DIR / 'metrics.json',
        'snapshot': PROCESSED_DIR / 'phase_2_featured_snapshot.csv',
        'raw_snapshot': PROCESSED_DIR / 'phase_1_raw_snapshot.csv',
        'accent': '#1f7a73',
        'data_path': None,
        'run_key': 'latest',
    },
    'CICIDS-2017': {
        'metrics': REAL_RUNS_DIR / 'cicids2017_metrics.json',
        'snapshot': REAL_RUNS_DIR / 'cicids2017_phase2_snapshot.csv',
        'raw_snapshot': REAL_RUNS_DIR / 'cicids2017_phase1_snapshot.csv',
        'accent': '#2f6fed',
        'data_path': ROOT / 'data' / 'raw' / 'cicids2017_machine_learning_0000.parquet',
        'run_key': 'cicids2017',
    },
    'CICIDS-2018': {
        'metrics': REAL_RUNS_DIR / 'cicids2018_metrics.json',
        'snapshot': REAL_RUNS_DIR / 'cicids2018_phase2_snapshot.csv',
        'raw_snapshot': REAL_RUNS_DIR / 'cicids2018_phase1_snapshot.csv',
        'accent': '#d17b0f',
        'data_path': ROOT / 'data' / 'raw' / 'cicids2018_network_traffic_0000.parquet',
        'run_key': 'cicids2018',
    },
}

THEMES = {
    'Light': {
        'bg_start': '#ecf7ff',
        'bg_mid': '#e2f2ff',
        'bg_end': '#f4f9ff',
        'panel_a': 'rgba(255,255,255,0.88)',
        'panel_b': 'rgba(236,246,255,0.80)',
        'panel_solid': 'rgba(248,252,255,0.92)',
        'ink': '#0b2340',
        'muted': '#4f6b87',
        'plot_bg': 'rgba(248,252,255,0.90)',
        'shadow': 'rgba(16,52,96,0.14)',
        'glow_a': 'rgba(34,211,238,0.22)',
        'glow_b': 'rgba(37,99,235,0.20)',
        'glow_c': 'rgba(125,211,252,0.18)',
        'line': 'rgba(86,145,255,0.13)',
        'accent': '#0891b2',
        'accent_alt': '#2563eb',
        'accent_warm': '#38bdf8',
        'accent_soft': '#60a5fa',
        'danger': '#ef4444',
        'surface_text': '#102a43',
        'sidebar_a': 'rgba(246,251,255,0.94)',
        'sidebar_b': 'rgba(228,241,255,0.92)',
        'chip_bg': 'rgba(11,145,178,0.10)',
        'chip_border': 'rgba(37,99,235,0.18)',
    },
    'Dark': {
        'bg_start': '#030b16',
        'bg_mid': '#08182d',
        'bg_end': '#0d1f3d',
        'panel_a': 'rgba(7,17,34,0.94)',
        'panel_b': 'rgba(11,27,49,0.90)',
        'panel_solid': 'rgba(10,22,42,0.92)',
        'ink': '#f3f9ff',
        'muted': '#b1c3d9',
        'plot_bg': 'rgba(8,18,36,0.90)',
        'shadow': 'rgba(0,0,0,0.46)',
        'glow_a': 'rgba(34,211,238,0.18)',
        'glow_b': 'rgba(56,189,248,0.14)',
        'glow_c': 'rgba(96,165,250,0.12)',
        'line': 'rgba(87,157,255,0.12)',
        'accent': '#67e8f9',
        'accent_alt': '#7dd3fc',
        'accent_warm': '#93c5fd',
        'accent_soft': '#60a5fa',
        'danger': '#fb7185',
        'surface_text': '#eaf4ff',
        'sidebar_a': 'rgba(5,15,30,0.96)',
        'sidebar_b': 'rgba(9,23,44,0.94)',
        'chip_bg': 'rgba(103,232,249,0.12)',
        'chip_border': 'rgba(125,211,252,0.18)',
    },
}


DEFAULT_METRICS = {
    'f1': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'roc_auc': 0.0,
    'latency_ms_avg': 0.0,
    'latency_ms_p99': 0.0,
    'confusion_matrix': [[0, 0], [0, 0]],
    'classification_report': 'No run loaded.',
}

NAV_HINT = ['Overview', 'Datasets & Training', 'Evaluation Analytics', 'Reports & Admin']


def bootstrap_state() -> None:
    theme_from_query = str(st.query_params.get('theme', '') or '')
    dataset_from_query = str(st.query_params.get('dataset', '') or '')
    st.session_state.setdefault('theme_name', 'Light')
    st.session_state.setdefault('selected_dataset', 'CICIDS-2017')
    if theme_from_query in THEMES and st.session_state.get('theme_name') != theme_from_query:
        st.session_state['theme_name'] = theme_from_query
    if dataset_from_query in DATASET_OPTIONS and st.session_state.get('selected_dataset') != dataset_from_query:
        st.session_state['selected_dataset'] = dataset_from_query
    st.session_state.setdefault('compare_mode', True)
    st.session_state.setdefault('auto_refresh', False)
    st.session_state.setdefault('refresh_seconds', 10)
    st.session_state.setdefault('max_rows', 20000)
    st.session_state.setdefault('run_message', '')
    st.session_state.setdefault('run_success', True)
    st.session_state.setdefault('toasts', [])
    st.session_state.setdefault('presentation_mode', True)
    st.session_state.setdefault('presentation_slide', 0)


def current_theme() -> dict[str, str]:
    return THEMES[st.session_state.get('theme_name', 'Light')]


def selected_config() -> dict[str, Any]:
    return DATASET_OPTIONS[st.session_state.get('selected_dataset', 'CICIDS-2017')]


def apply_theme() -> dict[str, str]:
    theme = current_theme()
    st.markdown(
        f"""
        <style>
        :root {{
          --ink: {theme['ink']};
          --muted: {theme['muted']};
          --accent: {theme['accent']};
          --accent-alt: {theme['accent_alt']};
          --accent-warm: {theme['accent_warm']};
          --line: {theme['line']};
          --chip-bg: {theme['chip_bg']};
          --chip-border: {theme['chip_border']};
        }}
        html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label, input, textarea, button {{ font-family: "Segoe UI", "Manrope", "Inter", "Arial", sans-serif; }}
        .stApp {{
          background:
            radial-gradient(circle at 12% 12%, {theme['glow_a']}, transparent 24%),
            radial-gradient(circle at 84% 10%, {theme['glow_c']}, transparent 20%),
            radial-gradient(circle at 54% 84%, {theme['glow_b']}, transparent 22%),
            linear-gradient(180deg, {theme['bg_start']} 0%, {theme['bg_mid']} 46%, {theme['bg_end']} 100%);
          color: var(--ink);
          font-family: "Manrope", sans-serif;
        }}
        .stApp::before {{
          content: '';
          position: fixed;
          inset: 0;
          pointer-events: none;
          background-image:
            linear-gradient(var(--line) 1px, transparent 1px),
            linear-gradient(90deg, var(--line) 1px, transparent 1px);
          background-size: 34px 34px;
          mask-image: linear-gradient(180deg, rgba(0,0,0,0.24), transparent 84%);
          opacity: 0.55;
          z-index: 0;
        }}
        .main .block-container {{
          position: relative;
          z-index: 1;
          max-width: 1420px;
          padding-top: 1.1rem;
          padding-bottom: 2rem;
          font-size: 1.05rem;
        }}
        [data-testid="stSidebar"] > div:first-child {{
          background:
            radial-gradient(circle at top left, {theme['glow_a']}, transparent 28%),
            linear-gradient(180deg, {theme['sidebar_a']}, {theme['sidebar_b']});
          border-right: 1px solid rgba(140, 191, 255, 0.14);
        }}
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox div,
        [data-testid="stSidebar"] .stSlider div,
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="radio"] *,
        [data-testid="stSidebar"] [data-baseweb="checkbox"] *,
        [data-testid="stSidebar"] [data-baseweb="slider"] *,
        [data-testid="stSidebar"] [data-baseweb="popover"] *,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] small {{
          color: var(--ink) !important;
          fill: var(--ink) !important;
          -webkit-text-fill-color: var(--ink) !important;
          font-size: 1rem !important;
        }}
        [data-testid="stSidebar"] svg {{ fill: var(--ink) !important; }}
        .hero, .shell, .report-shell, div[data-testid="stMetric"], .stTabs [data-baseweb="tab"], div[data-testid="stDataFrame"], div[data-testid="stAlert"] {{
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
        }}
        .hero, .shell, .report-shell {{
          position: relative;
          overflow: hidden;
          background: linear-gradient(145deg, {theme['panel_a']}, {theme['panel_b']});
          border: 1px solid rgba(127, 183, 255, 0.16);
          border-radius: 30px;
          box-shadow: 0 26px 60px {theme['shadow']}, inset 0 1px 0 rgba(255,255,255,0.08);
        }}
        .hero::before, .shell::before, .report-shell::before {{
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.09) 18%, transparent 38%);
          pointer-events: none;
        }}
        .hero {{ padding: 30px 32px; margin-bottom: 20px; }}
        .hero h1 {{
          font-family: "Space Grotesk", sans-serif;
          margin: 0;
          font-size: clamp(2.45rem, 3.8vw, 4rem);
          line-height: 1.04;
          letter-spacing: -0.04em;
          color: var(--ink);
          text-shadow: 0 0 28px rgba(56,189,248,0.08);
        }}
        .hero p {{
          color: var(--muted);
          margin: 14px 0 0 0;
          max-width: 1020px;
          line-height: 1.82;
          font-size: 1.08rem;
        }}
        .section-label {{
          color: var(--accent);
          font-size: 0.82rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          margin-bottom: 10px;
        }}
        .shell, .report-shell {{ padding: 24px; margin-bottom: 18px; font-size: 1.03rem; }}
        .shell h3, .shell h4, .shell strong, .report-shell h3 {{ color: var(--ink); }}
        .shell code, .report-shell code {{
          background: var(--chip-bg);
          border: 1px solid var(--chip-border);
          border-radius: 10px;
          padding: 0.1rem 0.45rem;
          color: var(--accent-alt);
        }}
        .phase-card {{
          border-radius: 22px;
          padding: 15px 17px;
          margin: 12px 0;
          border: 1px solid rgba(102, 171, 255, 0.18);
          text-align: center;
          box-shadow: 0 12px 24px rgba(15, 42, 79, 0.10);
        }}
        .phase-title {{ font-weight: 800; font-size: 1.12rem; margin-bottom: 4px; color: var(--ink); }}
        .phase-sub {{ color: var(--muted); font-size: 0.93rem; }}
        div[data-testid="stMetric"] {{
          background: linear-gradient(180deg, {theme['panel_solid']}, rgba(255,255,255,0.04));
          padding: 16px 18px;
          border-radius: 22px;
          border: 1px solid rgba(112, 180, 255, 0.14);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 14px 30px rgba(6, 22, 45, 0.10);
        }}
        div[data-testid="stMetric"] label {{ color: var(--muted) !important; font-size: 0.98rem !important; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.8rem !important; color: var(--ink) !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 10px; flex-wrap: wrap; }}
        .stTabs [data-baseweb="tab"] {{
          background: linear-gradient(180deg, {theme['panel_solid']}, rgba(255,255,255,0.03));
          border: 1px solid rgba(125, 182, 255, 0.14);
          border-radius: 16px;
          padding: 11px 16px;
          color: var(--ink);
          font-weight: 700;
        }}
        .stTabs [aria-selected="true"] {{
          box-shadow: 0 0 0 1px rgba(37,99,235,0.16), 0 0 24px rgba(96,165,250,0.18);
          color: var(--accent-alt);
        }}
        .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] button {{
          border-radius: 999px;
          border: 1px solid rgba(108, 188, 255, 0.28);
          background: linear-gradient(135deg, var(--accent-alt), var(--accent));
          color: #f7fbff;
          font-weight: 800;
          font-size: 1.02rem;
          min-height: 3.2rem;
          box-shadow: 0 0 0 1px rgba(255,255,255,0.05) inset, 0 14px 28px rgba(12, 72, 138, 0.22), 0 0 24px rgba(34,211,238,0.18);
          transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {{
          transform: translateY(-1px);
          border-color: rgba(190, 227, 255, 0.52);
          box-shadow: 0 16px 30px rgba(12, 72, 138, 0.22), 0 0 28px rgba(34,211,238,0.24);
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"], .stTextArea textarea {{
          border-radius: 16px !important;
          border: 1px solid rgba(110, 177, 255, 0.18) !important;
          background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04)) !important;
          color: var(--ink) !important;
          font-size: 1rem !important;
        }}
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label, .stToggle label {{
          color: var(--ink) !important;
          font-weight: 700 !important;
        }}
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stCaption, .stText, h1, h2, h3, h4, h5, h6, th, td {{
          color: var(--ink) !important;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"], [data-testid="stMetricValue"] * {{
          color: var(--ink) !important;
        }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {{
          color: var(--ink) !important;
          background: transparent !important;
        }}
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input,
        .stTextArea textarea {{
          background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03)) !important;
          color: var(--ink) !important;
        }}
        div[data-baseweb="popover"],
        ul[role="listbox"] {{
          background: {theme['panel_solid']} !important;
          border: 1px solid rgba(112, 180, 255, 0.18) !important;
          color: var(--ink) !important;
        }}
        [role="listbox"] *, [role="option"] * {{
          color: var(--ink) !important;
          -webkit-text-fill-color: var(--ink) !important;
        }}
        .stRadio label, .stCheckbox label, .stToggle label, .stSlider label, .stSelectbox label, .stMultiSelect label {{
          color: var(--ink) !important;
        }}
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
          border-radius: 22px;
          overflow: hidden;
          border: 1px solid rgba(112, 180, 255, 0.12);
          box-shadow: 0 14px 30px rgba(6, 22, 45, 0.10);
          background: linear-gradient(180deg, {theme['panel_solid']}, rgba(255,255,255,0.03));
        }}
        div[data-testid="stDataFrame"] *, div[data-testid="stTable"] * {{
          color: var(--ink) !important;
        }}
        div[data-testid="stAlert"] {{
          border-radius: 18px;
          border: 1px solid rgba(112, 180, 255, 0.12);
          box-shadow: 0 10px 24px rgba(8, 24, 42, 0.08);
        }}
        .js-plotly-plot .plotly text {{
          fill: var(--ink) !important;
        }}
        .stMarkdown p, .stMarkdown li, .stCaption, .stInfo, .stSuccess, .stWarning, .stError {{ font-size: 1rem; }}
        @media (max-width: 980px) {{ .hero {{ padding: 22px 20px; }} .hero h1 {{ font-size: 1.95rem; }} .shell {{ padding: 18px; }} }}
        @media (max-width: 640px) {{ .hero h1 {{ font-size: 1.52rem; }} .hero p {{ font-size: 0.97rem; }} .phase-title {{ font-size: 1rem; }} .phase-sub {{ font-size: 0.84rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return theme


def show_toasts() -> None:
    queued = st.session_state.get('toasts', [])
    if not queued:
        return
    for item in queued:
        st.toast(item)
    st.session_state['toasts'] = []


def queue_toast(message: str) -> None:
    st.session_state.setdefault('toasts', []).append(message)


def restore_authentication() -> bool:
    if st.session_state.get('authenticated') and st.session_state.get('user'):
        return True
    username = st.query_params.get('auth_user')
    if not username:
        return False
    user = fetch_user(str(username))
    if user is None:
        st.query_params.clear()
        return False
    st.session_state['authenticated'] = True
    st.session_state['user'] = user
    st.session_state['presentation_mode'] = False
    return True


def render_sidebar(page_title: str) -> None:
    st.sidebar.markdown(f'## {page_title}')
    theme_options = list(THEMES.keys())
    dataset_options = list(DATASET_OPTIONS.keys())
    current_theme_name = st.session_state.get('theme_name', 'Light')
    current_dataset_name = st.session_state.get('selected_dataset', 'CICIDS-2017')
    chosen_theme = st.sidebar.selectbox(
        'Theme',
        theme_options,
        index=theme_options.index(current_theme_name) if current_theme_name in theme_options else 0,
        key=f'theme_picker_{page_title}',
    )
    chosen_dataset = st.sidebar.selectbox(
        'Dataset',
        dataset_options,
        index=dataset_options.index(current_dataset_name) if current_dataset_name in dataset_options else 0,
        key=f'dataset_picker_{page_title}',
    )
    if chosen_theme != current_theme_name:
        st.session_state['theme_name'] = chosen_theme
        st.query_params['theme'] = chosen_theme
        st.rerun()
    if chosen_dataset != current_dataset_name:
        st.session_state['selected_dataset'] = chosen_dataset
        st.query_params['dataset'] = chosen_dataset
        st.rerun()
    st.sidebar.toggle('Compare datasets', key='compare_mode')
    st.sidebar.toggle('Auto refresh', key='auto_refresh')
    st.sidebar.select_slider('Refresh every', options=[5, 10, 15, 30, 60], key='refresh_seconds')
    st.sidebar.slider('Training rows', min_value=5000, max_value=50000, step=5000, key='max_rows')
    st.query_params['theme'] = st.session_state.get('theme_name', 'Light')
    st.query_params['dataset'] = st.session_state.get('selected_dataset', 'CICIDS-2017')
    user = st.session_state.get('user')
    if user:
        st.query_params['auth_user'] = user['username']
    st.sidebar.markdown('### Navigation')
    for item in NAV_HINT:
        st.sidebar.caption(item)
    user = st.session_state.get('user')
    if user:
        st.sidebar.success(f"Logged in as {user['username']}")
        if st.sidebar.button('Logout', use_container_width=True):
            st.session_state['authenticated'] = False
            st.session_state.pop('user', None)
            st.session_state['presentation_mode'] = True
            st.session_state['presentation_slide'] = 0
            st.query_params.clear()
            st.switch_page('app.py')


def require_authentication() -> None:
    if restore_authentication():
        return
    st.warning('Login required. Redirecting to home page.')
    st.switch_page('app.py')


def _cache_token(path: Path) -> tuple[str, float, int]:
    if not path.exists():
        return (str(path), 0.0, 0)
    stat = path.stat()
    return (str(path), stat.st_mtime, stat.st_size)


@st.cache_data(show_spinner=False)
def _cached_metrics(_token: tuple[str, float, int]) -> dict[str, Any]:
    path = Path(_token[0])
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return DEFAULT_METRICS.copy()


def load_metrics(path: Path) -> dict[str, Any]:
    return _cached_metrics(_cache_token(path))


@st.cache_data(show_spinner=False)
def _cached_frame(_token: tuple[str, float, int]) -> pd.DataFrame:
    path = Path(_token[0])
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_frame(path: Path) -> pd.DataFrame:
    raw = _cached_frame(_cache_token(path))
    if not raw.empty:
        numeric = raw.select_dtypes(include=['number']).copy()
        traffic = numeric.iloc[:, 0].abs().head(180).reset_index(drop=True) if not numeric.empty else pd.Series(np.random.default_rng(42).normal(850, 120, size=120))
        length = len(traffic)
        timestamps = pd.date_range(end=datetime.now(), periods=max(length, 1), freq='min')
        anomaly_source = numeric.iloc[:, 1].abs().head(length) if numeric.shape[1] > 1 else pd.Series(np.linspace(0.12, 0.96, length))
        congestion_source = numeric.iloc[:, 2].abs().head(length) if numeric.shape[1] > 2 else pd.Series(np.linspace(0.25, 0.88, length))
        anomaly = anomaly_source.to_numpy(dtype=float)
        if len(anomaly) and float(np.max(anomaly)) > 0:
            anomaly = anomaly / float(np.max(anomaly))
        congestion = congestion_source.to_numpy(dtype=float)
        if len(congestion) and float(np.max(congestion)) > 0:
            congestion = congestion / float(np.max(congestion))
        return pd.DataFrame({'timestamp': timestamps[:length], 'traffic': traffic.to_numpy(), 'anomaly_score': anomaly, 'congestion': congestion})

    timestamps = pd.date_range(end=datetime.now(), periods=120, freq='min')
    rng = np.random.default_rng(42)
    return pd.DataFrame({'timestamp': timestamps, 'traffic': rng.normal(850, 120, size=120).clip(150, None), 'anomaly_score': rng.uniform(0.1, 1.0, size=120), 'congestion': rng.uniform(0.2, 0.9, size=120)})


@st.cache_data(show_spinner=False)
def _cached_label_series(_token: tuple[str, float, int]) -> pd.Series:
    path = Path(_token[0])
    if not path.exists():
        return pd.Series(dtype='object')
    frame = pd.read_csv(path, usecols=lambda col: col.strip().lower() == 'label')
    if frame.empty or 'Label' not in frame.columns:
        return pd.Series(dtype='object')
    return frame['Label'].astype(str)


def load_label_distribution(path: Path) -> pd.DataFrame:
    labels = _cached_label_series(_cache_token(path))
    if labels.empty:
        return pd.DataFrame({'Label': ['Unavailable'], 'Count': [0]})
    counts = labels.value_counts().head(12).reset_index()
    counts.columns = ['Label', 'Count']
    return counts


def load_label_names(path: Path) -> list[str]:
    labels = _cached_label_series(_cache_token(path))
    if labels.empty:
        return []
    return sorted(labels.dropna().unique().tolist())


def parse_classification_report(report_text: str, label_names: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r'^\s*(.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$')
    label_map = {str(index): label for index, label in enumerate(label_names or [])}
    for line in report_text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        raw_label = match.group(1).strip()
        if raw_label.lower() in {'accuracy', 'macro avg', 'weighted avg'}:
            continue
        label = label_map.get(raw_label, raw_label)
        rows.append({'Label': label, 'Precision': float(match.group(2)), 'Recall': float(match.group(3)), 'F1': float(match.group(4)), 'Support': int(match.group(5))})
    return pd.DataFrame(rows)


def fetch_history_df(limit: int | None = None, dataset_name: str | None = None) -> pd.DataFrame:
    rows = fetch_run_history(limit=limit)
    if not rows:
        return pd.DataFrame(columns=['timestamp', 'dataset', 'max_rows', 'status', 'f1', 'roc_auc', 'precision_score', 'recall_score', 'latency_ms_avg', 'notes'])
    history = pd.DataFrame(rows)
    history['timestamp'] = pd.to_datetime(history['timestamp'])
    if dataset_name is not None:
        history = history[history['dataset'] == dataset_name]
    return history.sort_values('timestamp')


def persist_named_run(run_key: str, metrics: dict[str, Any]) -> None:
    if run_key == 'latest':
        return
    REAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (REAL_RUNS_DIR / f'{run_key}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    latest_phase1 = PROCESSED_DIR / 'phase_1_raw_snapshot.csv'
    latest_phase2 = PROCESSED_DIR / 'phase_2_featured_snapshot.csv'
    if latest_phase1.exists():
        shutil.copy2(latest_phase1, REAL_RUNS_DIR / f'{run_key}_phase1_snapshot.csv')
    if latest_phase2.exists():
        shutil.copy2(latest_phase2, REAL_RUNS_DIR / f'{run_key}_phase2_snapshot.csv')


def trigger_training(dataset_name: str, max_rows: int) -> tuple[bool, str]:
    from src.pipeline import run_all

    cfg = DATASET_OPTIONS[dataset_name]
    try:
        if cfg['data_path'] is None:
            artifacts = run_all(synthetic_rows=max_rows)
        else:
            artifacts = run_all(data_path=cfg['data_path'], max_rows=max_rows)
        persist_named_run(cfg['run_key'], artifacts.metrics)
        st.cache_data.clear()
        record_run(dataset_name, max_rows, 'success', artifacts.metrics)
        queue_toast(f'{dataset_name} training completed successfully')
        return True, f"Training completed for {dataset_name}. F1={artifacts.metrics['f1']:.3f}, ROC-AUC={artifacts.metrics['roc_auc']:.3f}"
    except Exception as exc:
        record_run(dataset_name, max_rows, 'failed', DEFAULT_METRICS, notes=str(exc))
        queue_toast(f'{dataset_name} training failed')
        return False, f'{dataset_name} training failed: {exc}'


def build_traffic_chart(frame: pd.DataFrame, accent: str, name: str, theme: dict[str, str]) -> go.Figure:
    timestamps = pd.to_datetime(frame['timestamp'])
    traffic = frame['traffic']
    anomaly_score = frame['anomaly_score']
    anomaly_mask = anomaly_score >= np.quantile(anomaly_score, 0.9)
    if accent.lower() == '#1f7a73':
        fill = 'rgba(31,122,115,0.12)'
    elif accent.lower() == '#2f6fed':
        fill = 'rgba(47,111,237,0.14)'
    elif accent.lower() == '#d17b0f':
        fill = 'rgba(209,123,15,0.14)'
    else:
        fill = 'rgba(96,165,250,0.14)'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=traffic, mode='lines', name=f'{name} Traffic', line=dict(color=accent, width=3), fill='tozeroy', fillcolor=fill))
    fig.add_trace(go.Scatter(x=timestamps[anomaly_mask], y=traffic[anomaly_mask], mode='markers', name='Anomalies', marker=dict(color=theme['danger'], size=10, line=dict(color=theme['surface_text'], width=2))))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        legend=dict(orientation='h', y=1.08),
        font=dict(color=theme['ink']),
        xaxis=dict(gridcolor=theme['line'], zeroline=False),
        yaxis=dict(gridcolor=theme['line'], zeroline=False),
    )
    return fig


def build_heatmap(metrics: dict[str, Any], theme: dict[str, str], label_names: list[str] | None = None) -> go.Figure:
    matrix = np.asarray(metrics.get('confusion_matrix', [[0, 0], [0, 0]]))
    if label_names and len(label_names) == matrix.shape[0] == matrix.shape[1]:
        x_labels = [f'Pred {label}' for label in label_names]
        y_labels = [f'Actual {label}' for label in label_names]
    else:
        x_labels = [f'Pred {idx}' for idx in range(matrix.shape[1])]
        y_labels = [f'Actual {idx}' for idx in range(matrix.shape[0])]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, '#dbeafe'], [0.45, '#60a5fa'], [0.75, '#2563eb'], [1, '#0f172a']],
            text=matrix,
            texttemplate='%{text}',
            textfont=dict(color=theme['surface_text'], size=14),
            showscale=False,
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        font=dict(color=theme['ink']),
        xaxis=dict(gridcolor=theme['line'], zeroline=False),
        yaxis=dict(gridcolor=theme['line'], zeroline=False),
    )
    fig.update_yaxes(autorange='reversed')
    return fig


def build_comparison_table() -> pd.DataFrame:
    rows = []
    for name, cfg in DATASET_OPTIONS.items():
        if not cfg['metrics'].exists():
            continue
        metrics = load_metrics(cfg['metrics'])
        rows.append({'Dataset': name, 'F1': round(metrics['f1'], 4), 'ROC-AUC': round(metrics['roc_auc'], 4), 'Precision': round(metrics['precision'], 4), 'Recall': round(metrics['recall'], 4), 'Avg Latency (ms)': round(metrics['latency_ms_avg'], 4)})
    return pd.DataFrame(rows)


def build_comparison_chart(comparison_df: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for metric, color in [('F1', theme['accent_alt']), ('ROC-AUC', theme['accent']), ('Recall', theme['accent_warm'])]:
        fig.add_trace(go.Bar(x=comparison_df['Dataset'], y=comparison_df[metric], name=metric, marker_color=color, text=comparison_df[metric], textposition='outside'))
    fig.update_layout(
        barmode='group',
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        yaxis_range=[0, 1.1],
        font=dict(color=theme['ink']),
        xaxis=dict(gridcolor=theme['line'], zeroline=False),
        yaxis=dict(gridcolor=theme['line'], zeroline=False),
    )
    return fig


def build_attack_breakdown_chart(label_df: pd.DataFrame, accent: str, theme: dict[str, str]) -> go.Figure:
    fig = go.Figure(data=[go.Bar(x=label_df['Count'], y=label_df['Label'], orientation='h', marker=dict(color=accent), text=label_df['Count'], textposition='outside')])
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=30, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        yaxis={'categoryorder': 'total ascending', 'gridcolor': theme['line'], 'zeroline': False},
        xaxis=dict(gridcolor=theme['line'], zeroline=False),
        font=dict(color=theme['ink']),
    )
    return fig


def build_history_chart(history_df: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    if history_df.empty:
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=theme['plot_bg'],
            font=dict(color=theme['ink']),
        )
        return fig
    fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['f1'], mode='lines+markers', name='F1 Score', line=dict(color=theme['accent_alt'], width=3)))
    fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['roc_auc'], mode='lines+markers', name='ROC-AUC', line=dict(color=theme['accent'], dash='dot', width=3)))
    fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['precision_score'], mode='lines+markers', name='Precision', line=dict(color=theme['accent_soft'], dash='dash', width=3)))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        yaxis_range=[0, 1.05],
        font=dict(color=theme['ink']),
        xaxis=dict(gridcolor=theme['line'], zeroline=False),
        yaxis=dict(gridcolor=theme['line'], zeroline=False),
    )
    return fig


def build_class_metrics_chart(class_df: pd.DataFrame, theme: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    if class_df.empty:
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=theme['plot_bg'],
            font=dict(color=theme['ink']),
        )
        return fig
    fig.add_trace(go.Bar(x=class_df['Label'], y=class_df['Precision'], name='Precision', marker_color=theme['accent_alt']))
    fig.add_trace(go.Bar(x=class_df['Label'], y=class_df['Recall'], name='Recall', marker_color=theme['accent_warm']))
    fig.add_trace(go.Bar(x=class_df['Label'], y=class_df['F1'], name='F1', marker_color=theme['accent']))
    fig.update_layout(
        barmode='group',
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=theme['plot_bg'],
        yaxis_range=[0, 1.05],
        font=dict(color=theme['ink']),
        xaxis=dict(gridcolor=theme['line'], zeroline=False),
        yaxis=dict(gridcolor=theme['line'], zeroline=False),
    )
    return fig


def build_report_dataframe(dataset_name: str, metrics: dict[str, Any], class_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = [
        {'Section': 'Summary', 'Metric': 'Dataset', 'Value': dataset_name},
        {'Section': 'Summary', 'Metric': 'F1', 'Value': metrics.get('f1', 0.0)},
        {'Section': 'Summary', 'Metric': 'ROC-AUC', 'Value': metrics.get('roc_auc', 0.0)},
        {'Section': 'Summary', 'Metric': 'Precision', 'Value': metrics.get('precision', 0.0)},
        {'Section': 'Summary', 'Metric': 'Recall', 'Value': metrics.get('recall', 0.0)},
        {'Section': 'Summary', 'Metric': 'Avg Latency (ms)', 'Value': metrics.get('latency_ms_avg', 0.0)},
        {'Section': 'Summary', 'Metric': 'P99 Latency (ms)', 'Value': metrics.get('latency_ms_p99', 0.0)},
    ]
    report_df = pd.DataFrame(summary_rows)
    if not class_df.empty:
        expanded = []
        for _, row in class_df.iterrows():
            expanded.append({'Section': 'Per-Class', 'Metric': f"{row['Label']} Precision", 'Value': row['Precision']})
            expanded.append({'Section': 'Per-Class', 'Metric': f"{row['Label']} Recall", 'Value': row['Recall']})
            expanded.append({'Section': 'Per-Class', 'Metric': f"{row['Label']} F1", 'Value': row['F1']})
            expanded.append({'Section': 'Per-Class', 'Metric': f"{row['Label']} Support", 'Value': row['Support']})
        report_df = pd.concat([report_df, pd.DataFrame(expanded)], ignore_index=True)
    return report_df


def build_pdf_bytes(lines: list[str]) -> bytes:
    safe_lines = [line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)') for line in lines]
    commands = ['BT', '/F1 12 Tf', '50 780 Td']
    for index, line in enumerate(safe_lines):
        if index == 0:
            commands.append(f'({line}) Tj')
        else:
            commands.extend(['0 -16 Td', f'({line}) Tj'])
    commands.append('ET')
    stream = '\n'.join(commands).encode('latin-1', errors='replace')
    objects = [
        b'1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n',
        b'2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n',
        b'3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n',
        b'4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n',
        f'5 0 obj << /Length {len(stream)} >> stream\n'.encode('latin-1') + stream + b'\nendstream endobj\n',
    ]
    pdf = bytearray(b'%PDF-1.4\n')
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref = len(pdf)
    pdf.extend(f'xref\n0 {len(objects)+1}\n'.encode('latin-1'))
    pdf.extend(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        pdf.extend(f'{offset:010d} 00000 n \n'.encode('latin-1'))
    pdf.extend(f'trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode('latin-1'))
    return bytes(pdf)


def build_report_downloads(dataset_name: str, metrics: dict[str, Any], class_df: pd.DataFrame) -> tuple[bytes, bytes, pd.DataFrame]:
    report_df = build_report_dataframe(dataset_name, metrics, class_df)
    csv_bytes = report_df.to_csv(index=False).encode('utf-8')
    lines = [
        'Network Anomaly Detection Report',
        f'Dataset: {dataset_name}',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'F1: {metrics.get("f1", 0.0):.4f}',
        f'ROC-AUC: {metrics.get("roc_auc", 0.0):.4f}',
        f'Precision: {metrics.get("precision", 0.0):.4f}',
        f'Recall: {metrics.get("recall", 0.0):.4f}',
    ]
    for _, row in class_df.iterrows():
        lines.append(f"{row['Label']}: precision={row['Precision']:.2f}, recall={row['Recall']:.2f}, f1={row['F1']:.2f}, support={row['Support']}")
    pdf_bytes = build_pdf_bytes(lines)
    return csv_bytes, pdf_bytes, report_df


def realtime_status(metrics: dict[str, Any]) -> tuple[str, str]:
    if metrics.get('roc_auc', 0.0) < 0.9:
        return 'warning', 'Model confidence has dipped below the target threshold.'
    if metrics.get('latency_ms_p99', 0.0) > 5.0:
        return 'warning', 'Tail latency is rising. Review the latest model run.'
    return 'success', 'Monitoring status healthy. Models are responding within target range.'




