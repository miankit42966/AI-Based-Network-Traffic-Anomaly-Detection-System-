from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from dashboard.common import apply_theme, bootstrap_state, load_metrics, queue_toast, render_sidebar, restore_authentication, show_toasts
from dashboard.db import authenticate_user, fetch_users, init_database

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / 'docs' / 'project_architecture.svg'
LATEST = ROOT / 'data' / 'processed' / 'models' / 'metrics.json'
M17 = ROOT / 'data' / 'processed' / 'real_runs' / 'cicids2017_metrics.json'
M18 = ROOT / 'data' / 'processed' / 'real_runs' / 'cicids2018_metrics.json'
TITLES = ['Team Intro', 'Problem', 'Solution', 'S/W Stack', 'Website Flow', 'Architecture', 'Pipeline', 'Results', 'Entry Point']

st.set_page_config(page_title='Network Anomaly Detection', layout='wide')
init_database()
bootstrap_state()
restore_authentication()


def score(value: float) -> str:
    return f'{value:.3f}'


def read_svg() -> str:
    if not ARCH.exists():
        return '<div class="fallback">Architecture diagram not available.</div>'
    return ARCH.read_text(encoding='utf-8-sig')


def svg_data_uri() -> str | None:
    if not ARCH.exists():
        return None
    payload = ARCH.read_text(encoding='utf-8-sig').encode('utf-8')
    return 'data:image/svg+xml;base64,' + base64.b64encode(payload).decode('ascii')


def ppt_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
        [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stToolbar"] { display:none; }
        .stApp {
          background:
            radial-gradient(circle at 10% 10%, rgba(56,189,248,.18), transparent 24%),
            radial-gradient(circle at 88% 14%, rgba(147,197,253,.16), transparent 22%),
            radial-gradient(circle at 58% 82%, rgba(34,211,238,.14), transparent 24%),
            linear-gradient(145deg,#04101d,#0b1d3a 58%,#102a54);
          color:#eef7ff;
          font-family:'Manrope',sans-serif;
        }
        .main .block-container { max-width: 1420px; padding-top: .9rem; padding-bottom: 2rem; }
        .progress-shell { margin-bottom: 18px; }
        .progress-nav { height: 100%; position: relative; }
        .progress-nav button {
          white-space: pre-line !important;
          text-align: center !important;
          justify-content: center !important;
          align-items: center !important;
          padding: 14px 12px !important;
          border-radius: 20px !important;
          min-height: 88px !important;
          width: 100% !important;
          font-size: .9rem !important;
          line-height: 1.3 !important;
          background: rgba(10,24,46,.72) !important;
          border: 1px solid rgba(137,192,255,.16) !important;
          color: #ddecff !important;
          box-shadow: 0 12px 26px rgba(4,10,24,.28) !important;
        }
        .progress-nav.done button {
          background: linear-gradient(180deg, rgba(19,81,131,.88), rgba(10,33,66,.88)) !important;
          border-color: rgba(108,189,255,.34) !important;
        }
        .progress-nav.active button {
          transform: translateY(-4px) !important;
          background: linear-gradient(180deg, rgba(111,237,255,.96), rgba(30,122,205,.95)) !important;
          border: 2px solid rgba(235,248,255,.98) !important;
          color: #061220 !important;
          box-shadow: 0 0 0 1px rgba(255,255,255,.35), 0 22px 40px rgba(56,189,248,.30), 0 0 36px rgba(125,211,252,.30) !important;
        }
        .stage {
          position: relative;
          overflow: hidden;
          min-height: 76vh;
          padding: 36px;
          border-radius: 36px;
          background: linear-gradient(135deg, rgba(8,22,43,.95), rgba(10,18,35,.95));
          border: 1px solid rgba(173,211,255,.14);
          box-shadow: 0 30px 80px rgba(2,7,18,.52), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .stage.stage-compact {
          min-height: auto;
          padding-bottom: 24px;
        }
        .stage.stage-compact .copy { margin-bottom: 0; }
        .stage.stage-compact .mini { margin-top: 14px; }
        .stage::before, .stage::after {
          content:'';
          position:absolute;
          border-radius:999px;
          filter:blur(8px);
          animation: float 14s ease-in-out infinite;
        }
        .stage::before {
          width: 320px; height: 320px; top: -80px; right: -40px;
          background: radial-gradient(circle, rgba(56,189,248,.24), transparent 70%);
        }
        .stage::after {
          width: 340px; height: 340px; left: -110px; bottom: -120px;
          background: radial-gradient(circle, rgba(125,211,252,.18), transparent 72%);
          animation-delay: -5s;
        }
        .grid { position: relative; z-index: 1; display: grid; grid-template-columns: 1.2fr .96fr; gap: 26px; align-items: start; }
        .grid.one { grid-template-columns: 1fr; }
        .grid.two { grid-template-columns: repeat(2,minmax(0,1fr)); }
        .grid.three { grid-template-columns: repeat(3,minmax(0,1fr)); }
        .grid.four { grid-template-columns: repeat(4,minmax(0,1fr)); }
        .eyebrow {
          display:inline-flex; padding:8px 14px; border-radius:999px; font-size:.8rem; letter-spacing:.14em; text-transform:uppercase;
          color:#a9ddff; background:rgba(80,141,255,.12); border:1px solid rgba(124,183,255,.18); margin-bottom:16px;
        }
        h1 {
          font-family:'Space Grotesk',sans-serif;
          font-size: clamp(2.55rem, 4.2vw, 5rem);
          line-height: 1.02;
          margin: 0;
          max-width: 940px;
          letter-spacing: -.04em;
        }
        .copy {
          margin-top: 18px;
          max-width: 920px;
          color: #cfdef0;
          font-size: 1.12rem;
          line-height: 1.84;
        }
        .line { height: 4px; border-radius: 999px; background: linear-gradient(90deg,#4cd0ff,#9ccaff,#7dd3fc); margin: 18px 0; }
        .badges, .mini { display:flex; flex-wrap:wrap; gap:12px; margin-top:22px; }
        .badge {
          padding: 10px 16px;
          border-radius: 999px;
          background: rgba(255,255,255,.06);
          border: 1px solid rgba(190,220,255,.14);
          color: #eef7ff;
          font-size: .95rem;
        }
        .card, .step, .metric {
          background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
          border: 1px solid rgba(191,222,255,.15);
          border-radius: 26px;
          padding: 22px;
          backdrop-filter: blur(16px);
          box-shadow: 0 18px 38px rgba(3,8,20,.25);
        }
        .title { font-size: 1.16rem; font-weight: 800; margin-bottom: 10px; color: #f8fbff; }
        .text, .metric-note { color: #d0dff0; font-size: 1.02rem; line-height: 1.78; }
        .team { display:grid; gap:12px; margin-top:16px; }
        .member { display:flex; justify-content:space-between; gap:12px; padding:15px 16px; border-radius:18px; background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.08); }
        .member strong { color:#f7fbff; font-size:1rem; }
        .member span { color:#b1d0f0; font-size:.95rem; }
        .ribbon { display:flex; justify-content:space-between; gap:18px; margin-top:22px; padding:18px 20px; border-radius:22px; background:linear-gradient(90deg,rgba(76,208,255,.14),rgba(125,211,252,.12)); border:1px solid rgba(169,221,255,.18); }
        .stat { font-size:.84rem; letter-spacing:.12em; text-transform:uppercase; color:#9fc1e3; font-weight:700; }
        .big { font-family:'Space Grotesk',sans-serif; font-size:2.1rem; color:#e7f8ff; margin:8px 0; }
        .step { min-height: 206px; position:relative; }
        .step::after { content:''; position:absolute; top:24px; right:-11px; width:22px; height:22px; border-radius:50%; background:linear-gradient(135deg,#4ccfff,#9ccaff); }
        .grid.four .step:nth-child(4)::after, .grid.four .step:nth-child(8)::after { display:none; }
        .no { width:44px; height:44px; border-radius:14px; display:inline-flex; align-items:center; justify-content:center; background:rgba(125,211,252,.16); border:1px solid rgba(166,224,255,.22); color:#dff5ff; font-weight:800; margin-bottom:14px; }
        .fallback { padding:28px; text-align:center; color:#dbeafe; border-radius:22px; background:rgba(255,255,255,.04); border:1px solid rgba(189,221,255,.12); }
        .stButton > button { width:100%; min-height:3.25rem; border-radius:999px; border:1px solid rgba(173,212,255,.24); background:linear-gradient(135deg,rgba(24,81,143,.94),rgba(13,40,73,.94)); color:#f6fbff; font-weight:800; font-size:1rem; box-shadow:0 16px 34px rgba(4,12,26,.34); }
        .stButton > button[kind="primary"] { background: linear-gradient(135deg,#7dd3fc,#2563eb); color:#05111f; border-color:rgba(226,244,255,.88); }
        .note { text-align:center; color:#bfd0e4; font-size:.96rem; margin-top:.55rem; }
        @keyframes float { 0%,100% { transform:translate3d(0,0,0) scale(1); } 50% { transform:translate3d(0,18px,0) scale(1.06); } }
        @media (max-width: 1180px) {
          .grid, .grid.three, .grid.four { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 980px) {
          .main .block-container { padding-top: .7rem; }
          .stage { padding: 26px; min-height: auto; }
          .grid, .grid.one, .grid.two, .grid.three, .grid.four { grid-template-columns: 1fr; }
          h1 { font-size: 2.15rem; }
          .copy { font-size: 1rem; line-height: 1.72; }
          .ribbon, .member { flex-direction: column; align-items: flex-start; }
        }
        @media (max-width: 640px) {
          .stage { padding: 18px; border-radius: 24px; }
          h1 { font-size: 1.72rem; }
          .copy { font-size: .96rem; }
          .progress-nav button { min-height: 72px !important; font-size: .76rem !important; padding: 10px 8px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def progress_html(i: int) -> str:
    parts = []
    for idx, title in enumerate(TITLES):
        cls = 'node'
        if idx < i:
            cls += ' done'
        if idx == i:
            cls += ' active'
        parts.append(f'<div class="{cls}"><div class="num">{idx + 1}</div><div class="label">{title}</div></div>')
    return f'<div class="progress">{"".join(parts)}</div>'


def render_progress_nav(i: int) -> None:
    rows = [TITLES[:5], TITLES[5:]]
    st.markdown('<div class="progress-shell">', unsafe_allow_html=True)
    start_idx = 0
    for row in rows:
        cols = st.columns(len(row), gap='small')
        for offset, title in enumerate(row):
            idx = start_idx + offset
            cls = 'progress-nav'
            if idx < i:
                cls += ' done'
            if idx == i:
                cls += ' active'
            with cols[offset]:
                st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                if st.button(f'{idx + 1}\n{title}', key=f'progress_{idx}', use_container_width=True):
                    st.session_state['presentation_slide'] = idx
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        start_idx += len(row)
    st.markdown('</div>', unsafe_allow_html=True)


def slides(latest: dict[str, float], m17: dict[str, float], m18: dict[str, float], svg: str, users: int) -> list[str]:
    return [
        f'''<section class="stage"><div class="grid"><div><div class="eyebrow">Industrial Showcase Presentation</div><h1>AI-Powered Network Anomaly Detection Platform</h1><div class="line"></div><div class="copy">This website now opens with a dynamic professional presentation so reviewers can understand the team, the problem, the technical workflow, and the business relevance before entering the live dashboard.</div><div class="badges"><span class="badge">CICIDS-2017 and CICIDS-2018</span><span class="badge">Isolation Forest + LSTM Autoencoder + XGBoost</span><span class="badge">Interactive monitoring, analytics, and reports</span><span class="badge">Current showcase F1: {score(float(latest.get('f1', 0.0)))}</span></div><div class="ribbon"><div><div class="stat">Project Positioning</div><strong>Cybersecurity analytics system for anomaly detection, congestion visibility, and reporting</strong></div><div><div class="stat">Seeded Demo Users</div><strong>{users}</strong></div></div></div><div class="card"><div class="title">Project Team</div><div class="text">Professional introduction for the members presenting this platform.</div><div class="team"><div class="member"><strong>Durgesh Mishra_30</strong><span>Project lead and presentation owner</span></div><div class="member"><strong>Priyanshu Mishra_31</strong><span>Data and pipeline collaboration</span></div><div class="member"><strong>Shruti</strong><span>Analysis and project coordination</span></div><div class="member"><strong>Harsh Nalawade</strong><span>System implementation support</span></div></div><div class="mini"><span class="badge">Presentation-first entry</span><span class="badge">Industrial review ready</span></div></div></div></section>''',
        '''<section class="stage"><div class="grid"><div><div class="eyebrow">Problem Statement</div><h1>Why intelligent anomaly detection is necessary in modern networks</h1><div class="copy">Large-scale networks generate high-volume traffic continuously. Manual observation and static signature rules are often too slow, too rigid, or too incomplete for fast-changing attack behavior. Teams need early detection, better visibility, and measurable outputs rather than only raw logs.</div><div class="badges"><span class="badge">Huge traffic volume</span><span class="badge">Rapidly changing threats</span><span class="badge">False positives and delay</span><span class="badge">Weak operational visibility</span></div></div><div class="grid two"><div class="card"><div class="title">Challenge 1</div><div class="text">Manual analysis cannot scale for continuous traffic monitoring.</div></div><div class="card"><div class="title">Challenge 2</div><div class="text">Rule-based systems may miss behavior-driven or zero-day style attacks.</div></div><div class="card"><div class="title">Challenge 3</div><div class="text">Security teams need scored alerts, charts, and historical evidence.</div></div><div class="card"><div class="title">Challenge 4</div><div class="text">Leaders also need congestion, latency, model quality, and reporting insight.</div></div></div></div></section>''',
        f'''<section class="stage"><div class="grid"><div><div class="eyebrow">Proposed Solution</div><h1>A complete AI security workflow with real-world usefulness</h1><div class="copy">The project solves the problem through end-to-end ingestion, preprocessing, feature engineering, ensemble machine learning, real-time scoring, alert generation, analytics, and report export. It behaves like a control room rather than a simple static dashboard.</div><div class="ribbon"><div><div class="stat">CICIDS-2017</div><strong>F1 {score(float(m17.get('f1', 0.0)))} | ROC-AUC {score(float(m17.get('roc_auc', 0.0)))}</strong></div><div><div class="stat">CICIDS-2018</div><strong>F1 {score(float(m18.get('f1', 0.0)))} | ROC-AUC {score(float(m18.get('roc_auc', 0.0)))}</strong></div></div></div><div class="grid two"><div class="card"><div class="title">Real-World Usefulness</div><div class="text">Helpful for SOC teams, network admins, managed security providers, and smart campus environments.</div></div><div class="card"><div class="title">Operational Value</div><div class="text">Supports faster anomaly visibility, earlier triage, and evidence-backed decision-making.</div></div><div class="card"><div class="title">Scalability</div><div class="text">The modular pipeline can be extended to more traffic sources and newer models.</div></div><div class="card"><div class="title">Industry Fit</div><div class="text">Relevant for enterprise networks, telecom labs, cybersecurity research, and showcase deployments.</div></div></div></section>''',
        '''<section class="stage"><div class="grid one"><div><div class="eyebrow">Software Layer</div><h1>What has been used in the software part of the website</h1><div class="copy">The system combines a Python analytical backend with a multi-page web interface, machine learning, deep learning, interactive charts, persistence, and testing support.</div></div><div class="grid three"><div class="card"><div class="title">Interface</div><div class="text">Streamlit, custom HTML and CSS presentation components, Plotly charts, responsive layout.</div></div><div class="card"><div class="title">Programming and Data</div><div class="text">Python, Pandas, NumPy, Joblib, parquet and CSV handling.</div></div><div class="card"><div class="title">ML Models</div><div class="text">Scikit-learn preprocessing, Isolation Forest, XGBoost with sklearn fallback.</div></div><div class="card"><div class="title">Deep Learning</div><div class="text">TensorFlow Keras LSTM autoencoder with sequence reconstruction scoring.</div></div><div class="card"><div class="title">Data Preparation</div><div class="text">Imputation, scaling, label encoding, one-hot encoding, SMOTE balancing.</div></div><div class="card"><div class="title">Persistence and Support</div><div class="text">SQLite for users and run history, Pytest for checks, PyShark and Scapy hooks for extension.</div></div></div></div></section>''',
        '''<section class="stage"><div class="grid one"><div><div class="eyebrow">Website Flow</div><h1>Complete website flow from first screen to live usage</h1><div class="copy">The user journey is structured so reviewers first understand the project and then move into the actual website for exploration, training, monitoring, analytics, and reporting.</div></div><div class="grid four"><div class="step"><div class="no">1</div><div class="title">Presentation Entry</div><div class="text">The website opens with this dynamic project deck.</div></div><div class="step"><div class="no">2</div><div class="title">Secure Login</div><div class="text">The viewer enters the live interface through authentication.</div></div><div class="step"><div class="no">3</div><div class="title">Dataset Selection</div><div class="text">Sidebar controls dataset, theme, refresh, and row budget.</div></div><div class="step"><div class="no">4</div><div class="title">Training Orchestrator</div><div class="text">The training page runs the complete pipeline on selected data.</div></div><div class="step"><div class="no">5</div><div class="title">Overview Monitoring</div><div class="text">Live KPIs, anomaly markers, heatmaps, and comparison charts.</div></div><div class="step"><div class="no">6</div><div class="title">Deep Evaluation</div><div class="text">Class-wise metrics, historical trends, and attack breakdown.</div></div><div class="step"><div class="no">7</div><div class="title">Reports and Admin</div><div class="text">CSV and PDF export, users, and run logs.</div></div><div class="step"><div class="no">8</div><div class="title">Decision Support</div><div class="text">Actionable evidence for project review and security analysis.</div></div></div></div></section>''',
        '''<section class="stage stage-compact"><div class="grid one"><div><div class="eyebrow">Architectural Diagram</div><h1>System architecture and data flow</h1><div class="copy">The architecture connects raw datasets, preprocessing, ensemble models, the detection engine, the dashboard, and SQLite persistence so the full pipeline remains modular and presentation-ready.</div><div class="mini"><span class="badge">Data -> preprocessing -> models -> detection -> dashboard -> persistence</span><span class="badge">Built for both academic clarity and practical demonstration</span></div></div></div></section>''',
        '''<section class="stage"><div class="grid one"><div><div class="eyebrow">Detailed Project Explanation</div><h1>Core pipeline in detail</h1><div class="copy">The platform is divided into clear phases so each part of the anomaly detection workflow can be demonstrated, evaluated, and improved independently.</div></div><div class="grid two"><div class="card"><div class="title">Phase 1: Data Collection</div><div class="text">Loads CICIDS-style CSV or parquet data and can generate synthetic fallback datasets for reliable demo execution.</div></div><div class="card"><div class="title">Phase 2: Preprocessing</div><div class="text">Cleans columns, removes duplicates, handles missing values, encodes labels, transforms categorical data, and scales numeric features.</div></div><div class="card"><div class="title">Phase 3: Feature Engineering</div><div class="text">Builds traffic-oriented features and sequence windows while removing high-cardinality identifiers such as timestamps and IP fields.</div></div><div class="card"><div class="title">Phase 4: Model Training</div><div class="text">Trains Isolation Forest, gradient boosting, and optional LSTM autoencoder so the final system uses both statistical and learned behavior.</div></div></div></div></section>''',
        f'''<section class="stage"><div class="grid one"><div><div class="eyebrow">Operations And Results</div><h1>How the project works after training</h1><div class="copy">After training, the analyzer extracts flow features, transforms them through the saved preprocessor, calculates ensemble scores, monitors congestion, creates alerts, and pushes results into the dashboard and reporting layers.</div></div><div class="grid two"><div class="card"><div class="title">Detection Engine</div><div class="text">Combines unsupervised scores, supervised attack scores, and optional LSTM sequence scores into one ensemble anomaly score.</div></div><div class="card"><div class="title">Alerts</div><div class="text">Generates timestamped anomaly and congestion alerts with severity labels for more practical output.</div></div><div class="card"><div class="title">Dashboard Modules</div><div class="text">Overview, Datasets and Training, Evaluation Analytics, and Reports and Admin provide complete presentation coverage.</div></div><div class="card"><div class="title">Persistence</div><div class="text">SQLite stores authentication and run history; CSV and PDF exports make the project review-friendly.</div></div></div><div class="grid two"><div class="metric"><div class="stat">Latest F1</div><div class="big">{score(float(latest.get('f1', 0.0)))}</div><div class="metric-note">Weighted result from saved pipeline artifacts.</div></div><div class="metric"><div class="stat">Latest Avg Latency</div><div class="big">{float(latest.get('latency_ms_avg', 0.0)):.2f} ms</div><div class="metric-note">Prediction latency from repeated test inference.</div></div><div class="metric"><div class="stat">CICIDS-2017 ROC-AUC</div><div class="big">{score(float(m17.get('roc_auc', 0.0)))}</div><div class="metric-note">Strong binary benchmark result.</div></div><div class="metric"><div class="stat">CICIDS-2018 ROC-AUC</div><div class="big">{score(float(m18.get('roc_auc', 0.0)))}</div><div class="metric-note">Multi-class benchmark result for the showcase story.</div></div></div></div></section>''',
        '''<section class="stage"><div class="grid"><div><div class="eyebrow">Live Website Entry</div><h1>The presentation ends here. The live website starts from here.</h1><div class="copy">This final slide acts as the official web entry point. After understanding the team, problem, solution, architecture, and working, the reviewer can now open the real dashboard experience.</div><div class="badges"><span class="badge">Click the launch button below to open the website</span><span class="badge">The next screen provides secure access to the full dashboard</span></div></div><div class="grid two"><div class="card"><div class="title">Demo Credentials</div><div class="text">Username: <strong>Durgesh</strong><br/>Password: <strong>MiniProject@2026</strong></div></div><div class="card"><div class="title">What opens next</div><div class="text">Login screen, dashboard home, overview analytics, training controls, evaluation modules, and report center.</div></div><div class="card"><div class="title">Presentation Advantage</div><div class="text">Industrial viewers receive complete context before interactive exploration begins.</div></div><div class="card"><div class="title">Positioning</div><div class="text">Academic methodology plus industry-style web showcase.</div></div></div></div></section>''',
    ]


def render_presentation() -> None:
    latest = load_metrics(LATEST)
    m17 = load_metrics(M17)
    m18 = load_metrics(M18)
    svg = read_svg()
    i = max(0, min(int(st.session_state.get('presentation_slide', 0)), len(TITLES) - 1))
    st.session_state['presentation_slide'] = i
    ppt_css()
    render_progress_nav(i)
    st.markdown(slides(latest, m17, m18, svg, len(fetch_users()))[i], unsafe_allow_html=True)
    if i == 5:
        svg_uri = svg_data_uri()
        if svg_uri is None:
            st.markdown('<div class="fallback">Architecture diagram not available.</div>', unsafe_allow_html=True)
        else:
            components.html(f"<div style='width:100%;background:rgba(255,255,255,0.04);padding:18px;border-radius:28px;border:1px solid rgba(189,221,255,0.12);overflow:hidden'><img src='{svg_uri}' style='width:100%;height:auto;display:block;border-radius:22px' /></div>", height=760, scrolling=False)
    if i == len(TITLES) - 1:
        c1, c2 = st.columns(2, gap='large')
        with c1:
            if st.button('Previous Slide', key='ppt_prev_last', use_container_width=True):
                st.session_state['presentation_slide'] = i - 1
                st.rerun()
        with c2:
            if st.button('Enter Live Website', key='ppt_enter', type='primary', use_container_width=True):
                st.session_state['presentation_mode'] = False
                queue_toast('Presentation completed. Opening the live website entry point.')
                st.rerun()
    else:
        a, b, c = st.columns([1, 1.15, 1], gap='large')
        with a:
            if st.button('Previous Slide', key=f'prev_{i}', use_container_width=True, disabled=i == 0):
                st.session_state['presentation_slide'] = max(0, i - 1)
                st.rerun()
        with b:
            if st.button('Skip To Website', key=f'skip_{i}', use_container_width=True):
                st.session_state['presentation_mode'] = False
                queue_toast('Presentation skipped. Opening the live website entry point.')
                st.rerun()
            st.markdown(f"<div class='note'>Slide {i + 1} of {len(TITLES)}. Continue to move through the complete project story.</div>", unsafe_allow_html=True)
        with c:
            if st.button('Next Slide', key=f'next_{i}', type='primary', use_container_width=True):
                st.session_state['presentation_slide'] = min(len(TITLES) - 1, i + 1)
                st.rerun()


def render_login() -> None:
    apply_theme()
    show_toasts()
    st.markdown("""<div class='hero'><div class='section-label'>Website Entry Point</div><h1>Network Anomaly Detection Platform</h1><p>The presentation deck has finished. You are now at the live website entry point, where the dashboard can be accessed securely for training, monitoring, analytics, and reporting.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class='shell'><div class='section-label'>Executive Access</div><h3 style='margin-top:0;'>Industrial-grade network visibility, model performance tracking, and report export in one secured workspace.</h3><p style='margin-bottom:0;'>This interface is tuned for project showcase use, so once you sign in the session stays on the dashboard until you explicitly log out.</p></div>""", unsafe_allow_html=True)
    left, _ = st.columns([0.28, 0.72])
    with left:
        if st.button('Open Presentation Again', use_container_width=True):
            st.session_state['presentation_mode'] = True
            st.session_state['presentation_slide'] = 0
            st.rerun()
    c1, c2 = st.columns([1.08, 0.92], gap='large')
    with c1:
        st.markdown('### Secure Login')
        with st.form('login_form'):
            username = st.text_input('Username', value='Durgesh')
            password = st.text_input('Password', type='password', value='MiniProject@2026')
            submitted = st.form_submit_button('Sign In To Website', use_container_width=True)
        if submitted:
            ok, user = authenticate_user(username, password)
            if ok and user is not None:
                st.session_state['authenticated'] = True
                st.session_state['user'] = user
                st.session_state['presentation_mode'] = False
                st.query_params['auth_user'] = user['username']
                queue_toast('Login successful')
                st.rerun()
            else:
                st.error('Invalid credentials. Please use the seeded demo account.')
    with c2:
        st.markdown('### Demo Credentials')
        st.code('Username: Durgesh\nPassword: MiniProject@2026')
        st.markdown('### What You Can Explore')
        st.markdown('- `Overview`: live KPIs, anomaly markers, heatmaps, and phase flow')
        st.markdown('- `Datasets & Training`: retraining controls and label distribution')
        st.markdown('- `Evaluation Analytics`: class-wise metrics and historical trends')
        st.markdown('- `Reports & Admin`: CSV/PDF export, users, and training logs')


def render_home() -> None:
    apply_theme()
    render_sidebar('Project Home')
    show_toasts()
    st.markdown("""<div class='hero'><div class='section-label'>Major Project Console</div><h1>Network Anomaly Detection Platform</h1><p>The system is now inside the authenticated website experience. Use the sidebar to move through monitoring, training, evaluation analytics, and reporting modules prepared for your showcase.</p></div>""", unsafe_allow_html=True)
    top1, top2, top3 = st.columns(3, gap='large')
    top1.metric('Security Mode', 'Active')
    top2.metric('Session State', 'Persistent')
    top3.metric('Presentation Re-entry', 'Logout Only')
    c1, c2 = st.columns([1.3, 1], gap='large')
    with c1:
        st.markdown('### App Pages')
        st.markdown('- `Overview`: live KPIs, traffic lens, phase flow, and comparison snapshot')
        st.markdown('- `Datasets & Training`: retraining controls, status, and dataset distribution')
        st.markdown('- `Evaluation Analytics`: class-wise metrics, heatmaps, and historical trends')
        st.markdown('- `Reports & Admin`: CSV/PDF export, user list, training logs, and summary')
    with c2:
        st.markdown('### Session')
        st.success(f"Signed in as {st.session_state['user']['username']}")
        st.caption('The presentation-first entry point will appear again after logout for repeat showcase use.')
    st.info('Use the sidebar to switch between pages. Refreshing the running site is enough to load the latest interface.')


if st.session_state.get('presentation_mode') and not st.session_state.get('authenticated'):

    render_presentation()
elif st.session_state.get('authenticated'):
    render_home()
else:
    render_login()

