# from pathlib import Path
# import io
# import warnings
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

# warnings.filterwarnings("ignore")

# # ============================================================
# # UAC-CareAI — POLISHED, DATA-ACCURATE SINGLE-FILE DASHBOARD
# # UI ONLY: reads existing outputs; does not retrain or modify ML.
# # ============================================================

# st.set_page_config(
#     page_title="UAC-CareAI | System Capacity Analytics",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# ROOT = Path(__file__).resolve().parent


# def first_existing(*paths):
#     for p in paths:
#         if p.exists():
#             return p
#     return paths[0]


# DATA_DIR = first_existing(ROOT / "Data" / "Processed", ROOT / "data" / "processed")
# REPORT_DIR = ROOT / "reports"
# MODEL_DIR = first_existing(ROOT / "Models", ROOT / "models")

# # ------------------------------------------------------------
# # Theme
# # ------------------------------------------------------------
# if "uac_dark" not in st.session_state:
#     st.session_state.uac_dark = True

# DARK = st.session_state.uac_dark

# if DARK:
#     BG, SIDEBAR, CARD, CARD2 = "#071522", "#061321", "#0d1e31", "#10243a"
#     BORDER, TEXT, MUTED, GRID = "#203850", "#eef5ff", "#91a5bc", "#20364c"
#     BLUE, PURPLE, TEAL, GREEN, ORANGE, RED = "#3b9cff", "#9b5cff", "#19c7b5", "#35d39b", "#ffb52e", "#ff5263"
# else:
#     BG, SIDEBAR, CARD, CARD2 = "#f4f7fb", "#0d2b49", "#ffffff", "#f8fbff"
#     BORDER, TEXT, MUTED, GRID = "#dce5ef", "#13233a", "#66788e", "#e2e9f1"
#     BLUE, PURPLE, TEAL, GREEN, ORANGE, RED = "#1677d2", "#7657d8", "#0aa89b", "#159a6d", "#d99100", "#dc4050"

# st.markdown(
#     f"""
#     <style>
#     .stApp {{background:{BG}; color:{TEXT};}}
#     [data-testid="stHeader"] {{background:{BG};}}
#     [data-testid="stSidebarNav"] {{display:none !important;}}
#     section[data-testid="stSidebar"] {{background:{SIDEBAR}; border-right:1px solid {BORDER};}}
#     section[data-testid="stSidebar"] * {{color:#edf5ff;}}
#     .block-container {{max-width:1720px; padding-top:.75rem; padding-bottom:2rem;}}
#     h1,h2,h3,h4 {{color:{TEXT} !important;}}
#     p, label, .stCaption {{color:{MUTED};}}
#     div[data-testid="stMetric"] {{
#         background:{CARD}; border:1px solid {BORDER}; border-radius:15px;
#         padding:14px 16px; min-height:108px;
#     }}
#     div[data-testid="stMetricLabel"] {{color:{MUTED} !important; font-size:.76rem;}}
#     div[data-testid="stMetricValue"] {{color:{TEXT} !important; font-weight:800;}}
#     div[data-testid="stMetricDelta"] {{font-size:.72rem;}}
#     div[data-testid="stVerticalBlockBorderWrapper"] > div {{
#         border-color:{BORDER} !important; border-radius:15px !important; background:{CARD};
#     }}
#     .stButton > button, .stDownloadButton > button {{
#         border-radius:9px; border:1px solid {BORDER}; background:{CARD2}; color:{TEXT};
#     }}
#     .nav-title {{font-size:1.5rem; font-weight:850; color:#f5f9ff;}}
#     .nav-sub {{font-size:.72rem; color:#9db4ca; line-height:1.35; margin-bottom:12px;}}
#     .kicker {{color:{BLUE}; font-size:.68rem; font-weight:850; letter-spacing:.13em; text-transform:uppercase;}}
#     .page-title {{font-size:2.05rem; font-weight:850; line-height:1.1; color:{TEXT};}}
#     .page-subtitle {{color:{MUTED}; font-size:.88rem; margin:2px 0 16px;}}
#     .small {{color:{MUTED}; font-size:.74rem;}}
#     .status-dot {{display:inline-block; width:8px; height:8px; border-radius:50%; background:{GREEN}; box-shadow:0 0 10px {GREEN}; margin-right:6px;}}
#     .pill {{display:inline-block; padding:4px 9px; border-radius:999px; font-size:.68rem; font-weight:700;}}
#     .alert-row,.insight-row {{padding:9px 0; border-bottom:1px solid {BORDER};}}
#     .alert-row:last-child,.insight-row:last-child {{border-bottom:0;}}
#     .section-label {{font-size:.72rem; font-weight:800; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em;}}
#     .footer {{color:{MUTED}; font-size:.7rem; text-align:center; padding:20px 0 4px;}}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ------------------------------------------------------------
# # File / data helpers
# # ------------------------------------------------------------
# def find_file(folder, *names):
#     if not folder.exists():
#         return None
#     exact = {p.name: p for p in folder.iterdir() if p.is_file()}
#     lower = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
#     for n in names:
#         if n in exact:
#             return exact[n]
#         if n.lower() in lower:
#             return lower[n.lower()]
#     return None


# def read_csv(folder, *names, dates=False):
#     p = find_file(folder, *names)
#     if not p:
#         return pd.DataFrame()
#     try:
#         d = pd.read_csv(p)
#         if dates:
#             dc = next((c for c in d.columns if norm(c) == "date"), None)
#             if dc and dc != "Date":
#                 d = d.rename(columns={dc: "Date"})
#             if "Date" in d.columns:
#                 d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
#         return d
#     except Exception:
#         return pd.DataFrame()


# def norm(x):
#     return "".join(ch.lower() for ch in str(x) if ch.isalnum())


# def find_col(df, *names):
#     if df.empty:
#         return None
#     lookup = {norm(c): c for c in df.columns}
#     for name in names:
#         if norm(name) in lookup:
#             return lookup[norm(name)]
#     return None


# def numeric(df, *names):
#     c = find_col(df, *names)
#     if c is None:
#         return pd.Series(np.nan, index=df.index, dtype=float)
#     s = df[c].astype(str).str.replace(",", "", regex=False)
#     return pd.to_numeric(s, errors="coerce")


# def canonicalize_main(d):
#     if d.empty:
#         return d

#     d = d.copy()
#     date_col = find_col(d, "Date")
#     if date_col and date_col != "Date":
#         d = d.rename(columns={date_col: "Date"})
#     d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
#     d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

#     # Prefer exact engineered features. Only fall back when unavailable.
#     d["Total Load"] = numeric(d, "Total System Load", "total_system_load", "Total Load")
#     cbp = numeric(d, "Children in CBP custody", "cbp_care", "CBP Custody", "cbp_custody")
#     hhs = numeric(d, "Children in HHS Care", "hhs_care", "HHS Care", "hhs load")
#     transfers = numeric(d, "Children transferred out of CBP custody", "transfers", "Transfers")
#     discharges = numeric(d, "Children discharged from HHS Care", "discharges", "Discharges")
#     net = numeric(d, "Net Intake Pressure", "net_intake_pressure", "Net Intake")
#     growth = numeric(d, "Daily Load Growth %", "load_growth_pct", "Load Growth")
#     vol = numeric(d, "Load Volatility 7D", "load_roll_std_7", "Load Volatility")

#     # If exact total load feature is absent, total system load is CBP + HHS.
#     if d["Total Load"].isna().all():
#         d["Total Load"] = cbp + hhs

#     d["CBP Custody"] = cbp
#     d["HHS Care"] = hhs
#     d["Transfers"] = transfers
#     d["Discharges"] = discharges
#     d["Net Intake"] = net
#     d["Load Growth"] = growth
#     d["Load Volatility"] = vol

#     return d


# @st.cache_data(show_spinner=False)
# def load_outputs():
#     main = read_csv(DATA_DIR, "uac_features.csv", "uac_cleaned.csv", dates=True)
#     anomaly = read_csv(DATA_DIR, "uac_anomaly_results.csv", "anomaly_results.csv", dates=True)
#     cluster = read_csv(DATA_DIR, "uac_operational_clusters.csv", "cluster_results.csv", dates=True)
#     stress = read_csv(DATA_DIR, "uac_stress_labels.csv", dates=True)

#     load_results = read_csv(REPORT_DIR, "load_model_results.csv", "regression_model_results.csv")
#     predictions = read_csv(REPORT_DIR, "load_predictions.csv")
#     stress_results = read_csv(REPORT_DIR, "stress_model_results.csv")
#     sarima = read_csv(REPORT_DIR, "sarima_forecast.csv", "forecast_30_day.csv", dates=True)
#     forecast_results = read_csv(REPORT_DIR, "forecast_results.csv")
#     shap = read_csv(REPORT_DIR, "shap_feature_importance.csv", "shap_global_importance.csv")
#     validation = read_csv(REPORT_DIR, "final_ml_validation_report.csv")

#     return main, anomaly, cluster, stress, load_results, predictions, stress_results, sarima, forecast_results, shap, validation


# (
#     raw_main,
#     anomaly_df,
#     cluster_df,
#     stress_df,
#     load_results,
#     predictions,
#     stress_results,
#     sarima_df,
#     forecast_results,
#     shap_df,
#     validation_df,
# ) = load_outputs()

# if raw_main.empty:
#     st.error("UAC dataset could not be loaded. Expected Data/Processed/uac_features.csv or Data/Processed/uac_cleaned.csv.")
#     st.stop()

# main = canonicalize_main(raw_main)

# # Merge saved outputs by date. Never overwrite core engineered values.
# for extra in (stress_df, anomaly_df, cluster_df):
#     if not extra.empty and "Date" in extra.columns:
#         cols = [c for c in extra.columns if c != "Date"]
#         for c in cols:
#             if c in main.columns:
#                 continue
#             temp = extra[["Date", c]].copy()
#             main = main.merge(temp, on="Date", how="left")

# # Exact saved stress output drives risk when available.
# stress_col = find_col(main, "Stress Level", "stress_level")
# stress_score_col = find_col(main, "Stress Score", "stress_score")
# if stress_col:
#     main["Risk"] = main[stress_col].astype(str).replace({"nan": "Unknown"})
# else:
#     main["Risk"] = "Unknown"
# main["Stress Score"] = numeric(main, "Stress Score", "stress_score") if stress_score_col else np.nan

# cluster_col = find_col(main, "Operational Cluster", "cluster", "Cluster")
# pattern_col = find_col(main, "Operational Pattern", "pattern", "Operational Pattern")
# main["Cluster"] = main[cluster_col] if cluster_col else "—"
# main["Pattern"] = main[pattern_col].astype(str) if pattern_col else "Operational pattern"

# anomaly_flag_col = find_col(main, "Anomaly Flag", "anomaly_flag")
# anomaly_status_col = find_col(main, "Anomaly Status", "anomaly_status")
# anomaly_score_col = find_col(main, "Anomaly Score", "anomaly_score")
# main["Anomaly"] = pd.to_numeric(main[anomaly_flag_col], errors="coerce").fillna(0) if anomaly_flag_col else 0
# main["Anomaly Status"] = main[anomaly_status_col].astype(str) if anomaly_status_col else "Normal"
# main["Anomaly Score"] = pd.to_numeric(main[anomaly_score_col], errors="coerce") if anomaly_score_col else np.nan

# # ------------------------------------------------------------
# # Navigation
# # ------------------------------------------------------------
# PAGES = [
#     ("⌂", "Overview"),
#     ("▥", "Capacity & Load"),
#     ("↗", "Forecasting"),
#     ("⚠", "Risk & Anomalies"),
#     ("◈", "Operational Patterns"),
#     ("✦", "Explainable AI"),
#     ("▤", "Model Performance"),
#     ("⇩", "Reports & Download"),
# ]
# if "page" not in st.session_state:
#     st.session_state.page = "Overview"

# with st.sidebar:
#     st.markdown("<div class='nav-title'>👥 UAC-CareAI</div>", unsafe_allow_html=True)
#     st.markdown("<div class='nav-sub'>System Capacity & Care-Load<br>Analytics Platform</div>", unsafe_allow_html=True)

#     for icon, label in PAGES:
#         active = st.session_state.page == label
#         if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True, type="primary" if active else "secondary"):
#             st.session_state.page = label
#             st.rerun()

#     st.divider()
#     st.markdown("<div class='section-label'>Data status</div>", unsafe_allow_html=True)
#     st.write(f"🟢 {len(main):,} reporting observations")
#     st.write(f"🟢 Latest data: {main.Date.max():%d %b %Y}")
#     st.write("🟢 ML outputs loaded")
#     st.write("🟢 Forecast output loaded" if not sarima_df.empty else "🟡 Forecast output missing")

#     st.divider()
#     st.markdown("<div class='section-label'>Appearance</div>", unsafe_allow_html=True)
#     toggle = st.toggle("Dark mode", value=st.session_state.uac_dark)
#     if toggle != st.session_state.uac_dark:
#         st.session_state.uac_dark = toggle
#         st.rerun()

#     st.divider()
#     st.caption("UAC-CareAI v1.0")
#     st.caption("ML • Forecasting • Risk • Explainability")

# # ------------------------------------------------------------
# # Executive header + global filters
# # ------------------------------------------------------------
# page = st.session_state.page
# min_date, max_date = main.Date.min().date(), main.Date.max().date()

# st.markdown(
#     f"""
#     <div class='top-header'>
#         <div>
#             <div class='top-brand'>UAC-CareAI</div>
#             <div class='top-sub'>System Capacity &amp; Care-Load Analytics Platform</div>
#         </div>
#         <div class='top-right'>
#             <div>Last Updated<br><b style='color:{TEXT}'>{max_date:%d %b %Y}</b></div>
#             <div class='live-badge'><span class='status-dot'></span>Live data</div>
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # Clear, labeled controls in a dedicated panel.
# with st.container(border=True):
#     st.markdown("<div class='filter-title'>Dashboard Controls</div>", unsafe_allow_html=True)
#     f1, f2, f3, f4, f5 = st.columns([1.15, 1.15, .9, 1.05, .8])
#     with f1:
#         st.markdown("<div class='filter-label'>Date from</div>", unsafe_allow_html=True)
#         start = st.date_input("Date from", min_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
#         st.markdown("<div class='filter-caption'>Start of selected period</div>", unsafe_allow_html=True)
#     with f2:
#         st.markdown("<div class='filter-label'>Date to</div>", unsafe_allow_html=True)
#         end = st.date_input("Date to", max_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
#         st.markdown("<div class='filter-caption'>End of selected period</div>", unsafe_allow_html=True)
#     with f3:
#         st.markdown("<div class='filter-label'>Granularity</div>", unsafe_allow_html=True)
#         granularity = st.selectbox("Granularity", ["Daily", "Weekly", "Monthly"], label_visibility="collapsed")
#         st.markdown("<div class='filter-caption'>Chart aggregation</div>", unsafe_allow_html=True)
#     with f4:
#         available_risks = sorted([x for x in main["Risk"].dropna().astype(str).unique() if x.lower() != "unknown"])
#         risk_options = ["All"] + available_risks
#         st.markdown("<div class='filter-label'>Risk level</div>", unsafe_allow_html=True)
#         risk_filter = st.selectbox("Risk level", risk_options, label_visibility="collapsed")
#         st.markdown("<div class='filter-caption'>Saved stress classification</div>", unsafe_allow_html=True)
#     with f5:
#         st.markdown("<div class='filter-label'>Data status</div>", unsafe_allow_html=True)
#         st.markdown("<div class='filter-status'><span></span>Live data loaded</div>", unsafe_allow_html=True)
#         st.markdown("<div class='filter-caption'>Existing project outputs</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# if pd.Timestamp(start) > pd.Timestamp(end):
#     st.error("The start date must be earlier than the end date.")
#     st.stop()

# base = main[main.Date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
# if risk_filter != "All":
#     base = base[base["Risk"].astype(str).str.casefold() == risk_filter.casefold()]

# if base.empty:
#     st.warning("No observations match the selected filters.")
#     st.stop()

# # ------------------------------------------------------------
# # General helpers
# # ------------------------------------------------------------
# def template():
#     return "plotly_dark" if DARK else "plotly_white"


# def style_fig(fig, height=340, legend=True):
#     fig.update_layout(
#         template=template(),
#         height=height,
#         margin=dict(l=16, r=16, t=42, b=24),
#         paper_bgcolor=CARD,
#         plot_bgcolor=CARD,
#         font=dict(color=TEXT, size=11),
#         hoverlabel=dict(bgcolor=CARD, font_color=TEXT),
#         legend=dict(orientation="h", y=-0.18) if legend else dict(),
#         xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
#         yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
#     )
#     return fig


# def header(title, subtitle, kicker):
#     st.markdown(f"<div class='kicker'>{kicker}</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


# def delta_pct(series):
#     s = pd.to_numeric(series, errors="coerce").dropna()
#     if len(s) < 2 or s.iloc[-2] == 0:
#         return None
#     return (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100


# def aggregate(data):
#     d = data.copy().set_index("Date")
#     agg = {
#         "Total Load": "mean",
#         "CBP Custody": "mean",
#         "HHS Care": "mean",
#         "Transfers": "sum",
#         "Discharges": "sum",
#         "Net Intake": "mean",
#     }
#     if granularity == "Weekly":
#         return d.resample("W").agg(agg).reset_index()
#     if granularity == "Monthly":
#         return d.resample("MS").agg(agg).reset_index()
#     return data.copy()


# def risk_color(level):
#     s = str(level).lower()
#     if "critical" in s or "high" in s:
#         return RED
#     if "moderate" in s or "elevated" in s:
#         return ORANGE
#     if "low" in s or "normal" in s:
#         return GREEN
#     return MUTED


# def latest_row(data):
#     return data.sort_values("Date").iloc[-1]


# def csv_bytes(data):
#     return data.to_csv(index=False).encode("utf-8")


# def saved_file_status():
#     outputs = [
#         ("uac_features.csv", DATA_DIR / "uac_features.csv"),
#         ("uac_anomaly_results.csv", DATA_DIR / "uac_anomaly_results.csv"),
#         ("uac_operational_clusters.csv", DATA_DIR / "uac_operational_clusters.csv"),
#         ("uac_stress_labels.csv", DATA_DIR / "uac_stress_labels.csv"),
#         ("load_model_results.csv", REPORT_DIR / "load_model_results.csv"),
#         ("load_predictions.csv", REPORT_DIR / "load_predictions.csv"),
#         ("stress_model_results.csv", REPORT_DIR / "stress_model_results.csv"),
#         ("sarima_forecast.csv", REPORT_DIR / "sarima_forecast.csv"),
#         ("forecast_results.csv", REPORT_DIR / "forecast_results.csv"),
#         ("shap_feature_importance.csv", REPORT_DIR / "shap_feature_importance.csv"),
#         ("final_ml_validation_report.csv", REPORT_DIR / "final_ml_validation_report.csv"),
#     ]
#     return pd.DataFrame([{"Artifact": n, "Status": "Available" if p.exists() else "Missing"} for n, p in outputs])


# # ============================================================
# # OVERVIEW
# # ============================================================
# if page == "Overview":
#     header("Overview", "Executive view of system capacity, care load, operational flows and saved risk signals.", "EXECUTIVE COMMAND CENTER")

#     latest = latest_row(base)
#     status = str(latest["Risk"])
#     status_color = risk_color(status)

#     k1, k2, k3, k4, k5 = st.columns(5)
#     k1.metric("Current System Load", f"{latest['Total Load']:,.0f}", f"{delta_pct(base['Total Load']):+.1f}%" if delta_pct(base["Total Load"]) is not None else None)
#     k2.metric("CBP Custody", f"{latest['CBP Custody']:,.0f}", f"{delta_pct(base['CBP Custody']):+.1f}%" if delta_pct(base["CBP Custody"]) is not None else None)
#     k3.metric("HHS Care", f"{latest['HHS Care']:,.0f}", f"{delta_pct(base['HHS Care']):+.1f}%" if delta_pct(base["HHS Care"]) is not None else None)
#     k4.metric("Net Intake Pressure", f"{latest['Net Intake']:+,.0f}", f"{delta_pct(base['Net Intake']):+.1f}%" if delta_pct(base["Net Intake"]) is not None else None)
#     k5.metric("Saved Risk Level", status, "From stress output" if status != "Unknown" else "Not available")

#     chart = aggregate(base)

#     a, b = st.columns([1.65, 1])
#     with a:
#         with st.container(border=True):
#             st.markdown("### 📈 System Load Trend")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=chart.Date, y=chart["Total Load"], name="Total System Load", mode="lines", line=dict(color=BLUE, width=3)))
#             fig.add_trace(go.Scatter(x=chart.Date, y=chart["CBP Custody"], name="CBP Custody", mode="lines", line=dict(color=PURPLE, width=1.8)))
#             fig.add_trace(go.Scatter(x=chart.Date, y=chart["HHS Care"], name="HHS Care", mode="lines", line=dict(color=TEAL, width=1.8)))
#             fig.update_layout(hovermode="x unified")
#             style_fig(fig, 405)
#             st.plotly_chart(fig, use_container_width=True, key="ov_load")

#     with b:
#         with st.container(border=True):
#             st.markdown("### 🚨 Recent Alerts")
#             recent = base.sort_values("Date", ascending=False)
#             rows = []
#             for _, r in recent.iterrows():
#                 if r["Anomaly"] == 1:
#                     rows.append(("Anomaly", r["Date"], RED, str(r["Anomaly Status"])))
#                 elif str(r["Risk"]).lower() in {"high", "critical"}:
#                     rows.append(("High stress level", r["Date"], risk_color(r["Risk"]), str(r["Risk"])))
#                 if len(rows) >= 6:
#                     break
#             if not rows:
#                 st.success("No saved high-priority alert condition in this filtered period.")
#             for title, dt, color, detail in rows:
#                 st.markdown(
#                     f"<div class='alert-row'><b style='color:{color}'>●</b> {title}<br><span class='small'>{dt:%d %b %Y} · {detail}</span></div>",
#                     unsafe_allow_html=True,
#                 )

#     a, b, c = st.columns([1, 1, 1.15])
#     with a:
#         with st.container(border=True):
#             st.markdown("### Risk / Stress Distribution")
#             r = base["Risk"].value_counts().rename_axis("Risk").reset_index(name="Observations")
#             if not r.empty:
#                 fig = px.pie(r, names="Risk", values="Observations", hole=.62, color="Risk",
#                              color_discrete_map={x: risk_color(x) for x in r["Risk"]})
#                 fig.update_traces(textinfo="percent", textposition="inside")
#                 style_fig(fig, 330)
#                 st.plotly_chart(fig, use_container_width=True, key="ov_risk")

#     with b:
#         with st.container(border=True):
#             st.markdown("### Transfers vs Discharges")
#             flow = aggregate(base)
#             fig = go.Figure()
#             fig.add_trace(go.Bar(x=flow.Date, y=flow["Transfers"], name="Transfers", marker_color=BLUE))
#             fig.add_trace(go.Bar(x=flow.Date, y=flow["Discharges"], name="Discharges", marker_color=TEAL))
#             fig.update_layout(barmode="group")
#             style_fig(fig, 330)
#             st.plotly_chart(fig, use_container_width=True, key="ov_flow")

#     with c:
#         with st.container(border=True):
#             st.markdown("### Key Takeaways")
#             d = delta_pct(base["Total Load"])
#             takeaways = [
#                 f"Latest system load is <b>{latest['Total Load']:,.0f}</b> children.",
#                 f"Saved stress level is <b>{status}</b>." if status != "Unknown" else "Saved stress level is not available for the current observation.",
#                 f"Selected-period anomaly observations: <b>{int(base['Anomaly'].sum()):,}</b>.",
#                 f"Latest net intake pressure is <b>{latest['Net Intake']:+,.0f}</b>.",
#             ]
#             if d is not None:
#                 takeaways.insert(1, f"Latest load changed <b>{abs(d):.1f}%</b> versus the prior observation.")
#             for item in takeaways:
#                 st.markdown(f"<div class='insight-row'>• {item}</div>", unsafe_allow_html=True)

#     with st.container(border=True):
#         st.markdown("### 🔮 Forecast Snapshot")
#         if sarima_df.empty:
#             st.info("No saved SARIMA output found.")
#         else:
#             f = sarima_df.copy()
#             fc = find_col(f, "Forecast Load", "Forecast", "forecast")
#             actual = find_col(f, "Actual Load", "Actual", "actual")
#             if fc:
#                 f[fc] = numeric(f, fc)
#                 f = f.dropna(subset=[fc]).sort_values("Date")
#                 if actual:
#                     f[actual] = numeric(f, actual)
#                 c1, c2, c3 = st.columns(3)
#                 c1.metric("Saved Forecast Rows", f"{len(f):,}")
#                 c2.metric("Latest Saved Forecast", f"{f.iloc[-1][fc]:,.0f}")
#                 c3.metric("Peak Saved Forecast", f"{f[fc].max():,.0f}")
#                 fig = go.Figure()
#                 if actual:
#                     fig.add_trace(go.Scatter(x=f.Date, y=f[actual], name="Actual", line=dict(color=BLUE, width=2)))
#                 fig.add_trace(go.Scatter(x=f.Date, y=f[fc], name="SARIMA", line=dict(color=PURPLE, width=2.5, dash="dash")))
#                 style_fig(fig, 300)
#                 st.plotly_chart(fig, use_container_width=True, key="ov_forecast")
#                 st.caption("The dashboard labels the saved file as forecast output; it does not assume validation rows are future projections.")

# # ============================================================
# # CAPACITY & LOAD
# # ============================================================
# elif page == "Capacity & Load":
#     header("Capacity & Load", "Interactive historical view of system load, custody, care and operational flows.", "CAPACITY ANALYTICS")

#     latest = latest_row(base)
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Average System Load", f"{base['Total Load'].mean():,.0f}")
#     c2.metric("Peak System Load", f"{base['Total Load'].max():,.0f}")
#     c3.metric("Transfers in Selection", f"{base['Transfers'].sum():,.0f}")
#     c4.metric("Discharges in Selection", f"{base['Discharges'].sum():,.0f}")

#     chart = aggregate(base)

#     a, b = st.columns([1.65, 1])
#     with a:
#         with st.container(border=True):
#             st.markdown("### Total System Load Over Time")
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=chart.Date, y=chart["Total Load"], name="Total Load", line=dict(color=BLUE, width=3)))
#             fig.add_trace(go.Scatter(x=chart.Date, y=chart["CBP Custody"], name="CBP Custody", line=dict(color=PURPLE, width=2)))
#             fig.add_trace(go.Scatter(x=chart.Date, y=chart["HHS Care"], name="HHS Care", line=dict(color=TEAL, width=2)))
#             fig.update_layout(hovermode="x unified")
#             style_fig(fig, 430)
#             st.plotly_chart(fig, use_container_width=True, key="cap_load")

#     with b:
#         with st.container(border=True):
#             st.markdown("### Current Load Composition")
#             vals = pd.DataFrame({"Component": ["CBP Custody", "HHS Care"], "Value": [latest["CBP Custody"], latest["HHS Care"]]})
#             fig = px.pie(vals, names="Component", values="Value", hole=.62,
#                          color="Component", color_discrete_map={"CBP Custody": PURPLE, "HHS Care": TEAL})
#             fig.update_traces(textinfo="label+percent")
#             style_fig(fig, 430)
#             st.plotly_chart(fig, use_container_width=True, key="cap_donut")

#     a, b = st.columns(2)
#     with a:
#         with st.container(border=True):
#             st.markdown("### Transfers vs Discharges")
#             fig = go.Figure()
#             fig.add_trace(go.Bar(x=chart.Date, y=chart["Transfers"], name="Transfers", marker_color=BLUE))
#             fig.add_trace(go.Bar(x=chart.Date, y=chart["Discharges"], name="Discharges", marker_color=TEAL))
#             fig.update_layout(barmode="group")
#             style_fig(fig, 360)
#             st.plotly_chart(fig, use_container_width=True, key="cap_flow")

#     with b:
#         with st.container(border=True):
#             st.markdown("### Net Intake Pressure")
#             fig = go.Figure()
#             fig.add_trace(go.Bar(x=chart.Date, y=chart["Net Intake"], name="Net Intake Pressure", marker_color=ORANGE))
#             style_fig(fig, 360)
#             st.plotly_chart(fig, use_container_width=True, key="cap_net")

#     with st.container(border=True):
#         st.markdown("### 7-Day / 14-Day Load Trend")
#         roll = base[["Date", "Total Load"]].sort_values("Date").copy()
#         roll["7-Day Average"] = roll["Total Load"].rolling(7, min_periods=1).mean()
#         roll["14-Day Average"] = roll["Total Load"].rolling(14, min_periods=1).mean()
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=roll.Date, y=roll["Total Load"], name="Total Load", line=dict(color=MUTED, width=1)))
#         fig.add_trace(go.Scatter(x=roll.Date, y=roll["7-Day Average"], name="7-Day Average", line=dict(color=BLUE, width=2.5)))
#         fig.add_trace(go.Scatter(x=roll.Date, y=roll["14-Day Average"], name="14-Day Average", line=dict(color=RED, width=2)))
#         style_fig(fig, 370)
#         st.plotly_chart(fig, use_container_width=True, key="cap_roll")

# # ============================================================
# # FORECASTING
# # ============================================================
# elif page == "Forecasting":
#     header("Forecasting", "Forward-looking analysis from the saved SARIMA/SARIMAX output. No forecast is invented in the dashboard.", "PREDICTIVE ANALYTICS")

#     if sarima_df.empty:
#         st.warning("Saved SARIMA forecast output was not found.")
#     else:
#         f = sarima_df.copy().dropna(subset=["Date"]).sort_values("Date")
#         fc = find_col(f, "Forecast Load", "Forecast", "forecast")
#         actual = find_col(f, "Actual Load", "Actual", "actual")
#         err = find_col(f, "Forecast Error", "Error", "error")

#         if not fc:
#             st.error("Forecast file exists, but no forecast column was found.")
#         else:
#             f[fc] = numeric(f, fc)
#             if actual:
#                 f[actual] = numeric(f, actual)
#             if err:
#                 f[err] = numeric(f, err)
#             f = f.dropna(subset=[fc])

#             has_validation_actual = actual is not None and f[actual].notna().any()

#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("Saved Forecast Rows", f"{len(f):,}")
#             c2.metric("Mean Forecast", f"{f[fc].mean():,.0f}")
#             c3.metric("Peak Forecast", f"{f[fc].max():,.0f}")
#             c4.metric("Output Type", "Validation output" if has_validation_actual else "Future output")

#             with st.container(border=True):
#                 st.markdown("### SARIMA Forecast Output")
#                 fig = go.Figure()
#                 if actual:
#                     fig.add_trace(go.Scatter(x=f.Date, y=f[actual], name="Actual Load", line=dict(color=BLUE, width=2)))
#                 fig.add_trace(go.Scatter(x=f.Date, y=f[fc], name="Forecast Load", line=dict(color=PURPLE, width=2.5, dash="dash")))
#                 lower = find_col(f, "Lower 95", "Lower_95", "Lower")
#                 upper = find_col(f, "Upper 95", "Upper_95", "Upper")
#                 if lower and upper:
#                     f[lower] = numeric(f, lower)
#                     f[upper] = numeric(f, upper)
#                     fig.add_trace(go.Scatter(x=f.Date, y=f[upper], line=dict(width=0), showlegend=False, hoverinfo="skip"))
#                     fig.add_trace(go.Scatter(x=f.Date, y=f[lower], fill="tonexty", line=dict(width=0), name="95% interval", hoverinfo="skip"))
#                 style_fig(fig, 455)
#                 st.plotly_chart(fig, use_container_width=True, key="fc_main")

#             if err and f[err].notna().any():
#                 e = f.dropna(subset=[err])
#                 mae = e[err].abs().mean()
#                 rmse = np.sqrt(np.mean(e[err] ** 2))
#                 a, b, c = st.columns(3)
#                 a.metric("Mean Absolute Error", f"{mae:,.2f}")
#                 b.metric("RMSE", f"{rmse:,.2f}")
#                 c.metric("Mean Error", f"{e[err].mean():,.2f}")
#                 with st.container(border=True):
#                     st.markdown("### Forecast Error")
#                     fig = px.bar(e, x="Date", y=err, color_discrete_sequence=[ORANGE])
#                     style_fig(fig, 330, legend=False)
#                     st.plotly_chart(fig, use_container_width=True, key="fc_error")

#             with st.container(border=True):
#                 st.markdown("### Saved Forecast Records")
#                 st.dataframe(f.tail(40).round(3), use_container_width=True, hide_index=True)

# # ============================================================
# # RISK & ANOMALIES
# # ============================================================
# elif page == "Risk & Anomalies":
#     header("Risk & Anomalies", "Review saved stress labels and anomaly-detection outputs for unusual operating conditions.", "RISK INTELLIGENCE")

#     latest = latest_row(base)
#     c1, c2, c3, c4 = st.columns(4)
#     c1.metric("Latest Saved Risk", str(latest["Risk"]))
#     c2.metric("Latest Stress Score", f"{latest['Stress Score']:.2f}" if pd.notna(latest["Stress Score"]) else "N/A")
#     c3.metric("Anomaly Observations", f"{int(base['Anomaly'].sum()):,}")
#     c4.metric("Latest Volatility", f"{latest['Load Volatility']:,.1f}" if pd.notna(latest["Load Volatility"]) else "N/A")

#     a, b = st.columns([1.5, 1])
#     with a:
#         with st.container(border=True):
#             st.markdown("### 🚨 Anomaly Timeline")
#             ad = base[base["Anomaly"] == 1].copy()
#             if ad.empty:
#                 st.success("No anomaly observations in the selected period.")
#             else:
#                 fig = px.scatter(
#                     ad, x="Date", y="Total Load", size="Anomaly Score",
#                     color="Risk", hover_data=["Net Intake", "Load Growth", "Anomaly Status"],
#                     color_discrete_map={x: risk_color(x) for x in ad["Risk"].astype(str).unique()},
#                 )
#                 style_fig(fig, 420)
#                 st.plotly_chart(fig, use_container_width=True, key="risk_anomaly")

#     with b:
#         with st.container(border=True):
#             st.markdown("### Stress Level Distribution")
#             r = base["Risk"].value_counts().rename_axis("Risk").reset_index(name="Observations")
#             fig = px.bar(r, x="Risk", y="Observations", color="Risk",
#                          color_discrete_map={x: risk_color(x) for x in r["Risk"]})
#             fig.update_layout(showlegend=False)
#             style_fig(fig, 420, legend=False)
#             st.plotly_chart(fig, use_container_width=True, key="risk_dist")

#     with st.container(border=True):
#         st.markdown("### Saved Anomaly / Risk Records")
#         cols = ["Date", "Total Load", "Net Intake", "Risk", "Stress Score", "Anomaly", "Anomaly Status", "Anomaly Score"]
#         cols = [c for c in cols if c in base.columns]
#         st.dataframe(base.sort_values("Date", ascending=False)[cols].head(100).round(3), use_container_width=True, hide_index=True)

# # ============================================================
# # OPERATIONAL PATTERNS
# # ============================================================
# elif page == "Operational Patterns":
#     header("Operational Patterns", "Explore the saved K-Means operational clusters and their observed load conditions.", "OPERATIONAL INTELLIGENCE")

#     if cluster_df.empty and main["Cluster"].eq("—").all():
#         st.warning("Operational clustering output was not found.")
#     else:
#         c1, c2, c3 = st.columns(3)
#         c1.metric("Clusters Observed", f"{main['Cluster'].nunique(dropna=True):,}")
#         c2.metric("Largest Cluster", str(main["Cluster"].value_counts().index[0]) if not main["Cluster"].empty else "—")
#         c3.metric("Selected Observations", f"{len(base):,}")

#         a, b = st.columns([1, 1.4])
#         with a:
#             with st.container(border=True):
#                 st.markdown("### Cluster Distribution")
#                 cc = base["Cluster"].astype(str).value_counts().rename_axis("Cluster").reset_index(name="Observations")
#                 fig = px.bar(cc, x="Cluster", y="Observations", color="Cluster")
#                 fig.update_layout(showlegend=False)
#                 style_fig(fig, 360, legend=False)
#                 st.plotly_chart(fig, use_container_width=True, key="cluster_dist")

#         with b:
#             with st.container(border=True):
#                 st.markdown("### Load vs Intake by Cluster")
#                 cd = base.copy()
#                 fig = px.scatter(cd, x="Net Intake", y="Total Load", color="Cluster",
#                                  hover_data=["Date", "CBP Custody", "HHS Care", "Pattern"])
#                 style_fig(fig, 360)
#                 st.plotly_chart(fig, use_container_width=True, key="cluster_scatter")

#         with st.container(border=True):
#             st.markdown("### Operational Pattern Summary")
#             summary = (
#                 base.groupby(["Cluster", "Pattern"], dropna=False)
#                 .agg(
#                     Observations=("Date", "count"),
#                     Avg_Load=("Total Load", "mean"),
#                     Avg_Net_Intake=("Net Intake", "mean"),
#                     Avg_CBP=("CBP Custody", "mean"),
#                     Avg_HHS=("HHS Care", "mean"),
#                 )
#                 .reset_index()
#                 .sort_values("Observations", ascending=False)
#             )
#             st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

# # ============================================================
# # EXPLAINABLE AI
# # ============================================================
# elif page == "Explainable AI":
#     header("Explainable AI", "Global SHAP feature influence for the saved load-prediction model.", "MODEL EXPLAINABILITY")

#     if shap_df.empty:
#         st.warning("SHAP feature-importance output was not found.")
#     else:
#         feature = find_col(shap_df, "Feature", "feature")
#         importance = find_col(shap_df, "Mean Absolute SHAP", "Mean_Absolute_SHAP", "mean_absolute_shap", "importance")
#         if not feature or not importance:
#             st.error("The SHAP report does not contain the expected feature and importance columns.")
#         else:
#             s = shap_df.copy()
#             s[importance] = numeric(s, importance)
#             s = s.dropna(subset=[importance]).sort_values(importance, ascending=False)
#             top = s.head(12).sort_values(importance)

#             a, b = st.columns([1.5, .75])
#             with a:
#                 with st.container(border=True):
#                     st.markdown("### Global Feature Influence")
#                     fig = px.bar(top, x=importance, y=feature, orientation="h")
#                     fig.update_traces(marker_color=PURPLE)
#                     style_fig(fig, 470, legend=False)
#                     st.plotly_chart(fig, use_container_width=True, key="shap")
#             with b:
#                 with st.container(border=True):
#                     st.markdown("### Explanation")
#                     st.metric("Top Feature", str(s.iloc[0][feature]))
#                     st.metric("Features Analysed", f"{len(s):,}")
#                     st.write("Mean absolute SHAP measures the average magnitude of each feature's contribution to the saved model predictions.")
#                     st.caption("This page explains the model; it does not retrain or alter it.")

#             with st.container(border=True):
#                 st.markdown("### SHAP Importance Table")
#                 st.dataframe(s.head(25).round(5), use_container_width=True, hide_index=True)

# # ============================================================
# # MODEL PERFORMANCE
# # ============================================================
# elif page == "Model Performance":
#     header("Model Performance", "Evaluate the saved regression and stress-classification results produced by the ML pipeline.", "MODEL GOVERNANCE")

#     if not load_results.empty:
#         model = find_col(load_results, "Model")
#         rmse = find_col(load_results, "RMSE")
#         mae = find_col(load_results, "MAE")
#         r2 = find_col(load_results, "R2", "R²")

#         if model and rmse:
#             lr = load_results.copy()
#             for c in [rmse, mae, r2]:
#                 if c:
#                     lr[c] = numeric(lr, c)
#             best = lr.dropna(subset=[rmse]).sort_values(rmse).iloc[0]

#             a, b, c, d = st.columns(4)
#             a.metric("Best Saved Model", str(best[model]))
#             b.metric("RMSE", f"{best[rmse]:.2f}")
#             c.metric("MAE", f"{best[mae]:.2f}" if mae else "N/A")
#             d.metric("R²", f"{best[r2]:.3f}" if r2 else "N/A")

#             x, y = st.columns(2)
#             with x:
#                 with st.container(border=True):
#                     fig = px.bar(lr.sort_values(rmse), x=model, y=rmse, text_auto=".2f")
#                     fig.update_traces(marker_color=BLUE)
#                     style_fig(fig, 350, legend=False)
#                     fig.update_layout(title="RMSE — Lower is Better")
#                     st.plotly_chart(fig, use_container_width=True, key="perf_rmse")
#             with y:
#                 with st.container(border=True):
#                     if r2:
#                         fig = px.bar(lr.sort_values(r2, ascending=False), x=model, y=r2, text_auto=".3f")
#                         fig.update_traces(marker_color=TEAL)
#                         style_fig(fig, 350, legend=False)
#                         fig.update_layout(title="R² — Higher is Better")
#                         st.plotly_chart(fig, use_container_width=True, key="perf_r2")

#             with st.container(border=True):
#                 st.markdown("### Regression Benchmark")
#                 st.dataframe(lr.round(4), use_container_width=True, hide_index=True)

#     if not predictions.empty:
#         date_c = find_col(predictions, "Date")
#         actual = find_col(predictions, "Actual Load", "Actual", "y_true")
#         pred = find_col(predictions, "Predicted Load", "Predicted", "Prediction", "y_pred")
#         if actual and pred:
#             p = predictions.copy()
#             xvals = pd.to_datetime(p[date_c], errors="coerce") if date_c else np.arange(len(p))
#             p[actual] = numeric(p, actual)
#             p[pred] = numeric(p, pred)
#             with st.container(border=True):
#                 st.markdown("### Actual vs Predicted Load")
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(x=xvals, y=p[actual], name="Actual", line=dict(color=BLUE, width=2.2)))
#                 fig.add_trace(go.Scatter(x=xvals, y=p[pred], name="Predicted", line=dict(color=PURPLE, width=2, dash="dash")))
#                 style_fig(fig, 400)
#                 st.plotly_chart(fig, use_container_width=True, key="perf_pred")

#     if not stress_results.empty:
#         model = find_col(stress_results, "Model")
#         f1 = find_col(stress_results, "F1", "F1 Score", "f1_score")
#         if model and f1:
#             sr = stress_results.copy()
#             sr[f1] = numeric(sr, f1)
#             with st.container(border=True):
#                 st.markdown("### Stress Classification Benchmark")
#                 fig = px.bar(sr, x=model, y=f1, text_auto=".3f")
#                 fig.update_traces(marker_color=PURPLE)
#                 style_fig(fig, 350, legend=False)
#                 st.plotly_chart(fig, use_container_width=True, key="perf_f1")
#                 st.dataframe(sr.round(4), use_container_width=True, hide_index=True)

#     with st.container(border=True):
#         st.markdown("### Saved AI Component Status")
#         status = [
#             ["Load Prediction", "Regression benchmark", "Available" if not load_results.empty else "Missing"],
#             ["Stress Classification", "Classification benchmark", "Available" if not stress_results.empty else "Missing"],
#             ["Anomaly Detection", "Isolation Forest output", "Available" if not anomaly_df.empty else "Missing"],
#             ["Operational Clustering", "K-Means output", "Available" if not cluster_df.empty else "Missing"],
#             ["Forecasting", "SARIMA/SARIMAX output", "Available" if not sarima_df.empty else "Missing"],
#             ["Explainability", "SHAP output", "Available" if not shap_df.empty else "Missing"],
#         ]
#         st.dataframe(pd.DataFrame(status, columns=["Component", "Evidence", "Status"]), use_container_width=True, hide_index=True)

# # ============================================================
# # REPORTS
# # ============================================================
# elif page == "Reports & Download":
#     header("Reports & Download", "Export the filtered evidence used by the dashboard and inspect artifact availability.", "PROJECT EVIDENCE")

#     c1, c2, c3 = st.columns(3)
#     c1.metric("Source Observations", f"{len(main):,}")
#     c2.metric("Full Date Coverage", f"{main.Date.min():%d %b %Y} → {main.Date.max():%d %b %Y}")
#     c3.metric("Filtered Observations", f"{len(base):,}")

#     with st.container(border=True):
#         st.markdown("### Export Current Selection")
#         st.download_button(
#             "⬇ Download filtered CSV",
#             data=csv_bytes(base),
#             file_name="uac_careai_filtered.csv",
#             mime="text/csv",
#             use_container_width=True,
#         )

#     with st.container(border=True):
#         st.markdown("### Analytical Artifact Status")
#         st.dataframe(saved_file_status(), use_container_width=True, hide_index=True)

#     with st.container(border=True):
#         st.markdown("### Filtered Data Preview")
#         st.dataframe(base.sort_values("Date", ascending=False).head(75).round(3), use_container_width=True, hide_index=True)

# st.markdown(
#     "<div class='footer'>UAC-CareAI • System Capacity & Care-Load Analytics • Existing ML outputs only</div>",
#     unsafe_allow_html=True,
# )



from pathlib import Path
import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# UAC-CareAI — POLISHED, DATA-ACCURATE SINGLE-FILE DASHBOARD
# UI ONLY: reads existing outputs; does not retrain or modify ML.
# ============================================================

st.set_page_config(
    page_title="UAC-CareAI | System Capacity Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent


def first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]


DATA_DIR = first_existing(ROOT / "Data" / "Processed", ROOT / "data" / "processed")
REPORT_DIR = ROOT / "reports"
MODEL_DIR = first_existing(ROOT / "Models", ROOT / "models")

# ------------------------------------------------------------
# Theme
# ------------------------------------------------------------
if "uac_dark" not in st.session_state:
    st.session_state.uac_dark = True

DARK = st.session_state.uac_dark

if DARK:
    BG, SIDEBAR, CARD, CARD2 = "#071522", "#061321", "#0d1e31", "#10243a"
    BORDER, TEXT, MUTED, GRID = "#203850", "#eef5ff", "#91a5bc", "#20364c"
    BLUE, PURPLE, TEAL, GREEN, ORANGE, RED = "#3b9cff", "#9b5cff", "#19c7b5", "#35d39b", "#ffb52e", "#ff5263"
else:
    BG, SIDEBAR, CARD, CARD2 = "#f4f7fb", "#0d2b49", "#ffffff", "#f8fbff"
    BORDER, TEXT, MUTED, GRID = "#dce5ef", "#13233a", "#66788e", "#e2e9f1"
    BLUE, PURPLE, TEAL, GREEN, ORANGE, RED = "#1677d2", "#7657d8", "#0aa89b", "#159a6d", "#d99100", "#dc4050"

st.markdown(
    f"""
    <style>
    .stApp {{background:{BG}; color:{TEXT};}}
    [data-testid="stHeader"] {{background:{BG};}}
    [data-testid="stSidebarNav"] {{display:none !important;}}
    section[data-testid="stSidebar"] {{background:{SIDEBAR}; border-right:1px solid {BORDER};}}
    section[data-testid="stSidebar"] * {{color:#edf5ff;}}
    .block-container {{max-width:1720px; padding-top:.75rem; padding-bottom:2rem;}}
    h1,h2,h3,h4 {{color:{TEXT} !important;}}
    p, label, .stCaption {{color:{MUTED};}}
    div[data-testid="stMetric"] {{
        background:{CARD}; border:1px solid {BORDER}; border-radius:15px;
        padding:14px 16px; min-height:108px;
    }}
    div[data-testid="stMetricLabel"] {{color:{MUTED} !important; font-size:.76rem;}}
    div[data-testid="stMetricValue"] {{color:{TEXT} !important; font-weight:800;}}
    div[data-testid="stMetricDelta"] {{font-size:.72rem;}}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        border-color:{BORDER} !important; border-radius:15px !important; background:{CARD};
    }}
    .stButton > button, .stDownloadButton > button {{
        border-radius:9px; border:1px solid {BORDER}; background:{CARD2}; color:{TEXT};
    }}
    .nav-title {{font-size:1.5rem; font-weight:850; color:#f5f9ff;}}
    .nav-sub {{font-size:.72rem; color:#9db4ca; line-height:1.35; margin-bottom:12px;}}
    .kicker {{color:{BLUE}; font-size:.68rem; font-weight:850; letter-spacing:.13em; text-transform:uppercase;}}
    .page-title {{font-size:2.05rem; font-weight:850; line-height:1.1; color:{TEXT};}}
    .page-subtitle {{color:{MUTED}; font-size:.88rem; margin:2px 0 16px;}}
    .small {{color:{MUTED}; font-size:.74rem;}}
    .status-dot {{display:inline-block; width:8px; height:8px; border-radius:50%; background:{GREEN}; box-shadow:0 0 10px {GREEN}; margin-right:6px;}}
    .pill {{display:inline-block; padding:4px 9px; border-radius:999px; font-size:.68rem; font-weight:700;}}
    .alert-row,.insight-row {{padding:9px 0; border-bottom:1px solid {BORDER};}}
    .alert-row:last-child,.insight-row:last-child {{border-bottom:0;}}
    .section-label {{font-size:.72rem; font-weight:800; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em;}}
    .footer {{color:{MUTED}; font-size:.7rem; text-align:center; padding:20px 0 4px;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# File / data helpers
# ------------------------------------------------------------
def find_file(folder, *names):
    if not folder.exists():
        return None
    exact = {p.name: p for p in folder.iterdir() if p.is_file()}
    lower = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    for n in names:
        if n in exact:
            return exact[n]
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def read_csv(folder, *names, dates=False):
    p = find_file(folder, *names)

    if p is None:
        return pd.DataFrame()

    last_error = None

    # Support normal UTF-8, UTF-8 with BOM, and legacy CSV exports.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            d = pd.read_csv(p, encoding=encoding)

            if dates:
                dc = next(
                    (c for c in d.columns if norm(c) == "date"),
                    None
                )

                if dc and dc != "Date":
                    d = d.rename(columns={dc: "Date"})

                if "Date" in d.columns:
                    d["Date"] = pd.to_datetime(
                        d["Date"],
                        errors="coerce"
                    )

            return d

        except Exception as e:
            last_error = e

    # Do not hide the real deployment error.
    st.error(
        f"Could not read `{p.as_posix()}`. "
        f"{type(last_error).__name__}: {last_error}"
    )
    return pd.DataFrame()


def norm(x):
    return "".join(ch.lower() for ch in str(x) if ch.isalnum())


def find_col(df, *names):
    if df.empty:
        return None
    lookup = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in lookup:
            return lookup[norm(name)]
    return None


def numeric(df, *names):
    c = find_col(df, *names)
    if c is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    s = df[c].astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def canonicalize_main(d):
    if d.empty:
        return d

    d = d.copy()
    date_col = find_col(d, "Date")
    if date_col and date_col != "Date":
        d = d.rename(columns={date_col: "Date"})
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Prefer exact engineered features. Only fall back when unavailable.
    d["Total Load"] = numeric(d, "Total System Load", "total_system_load", "Total Load")
    cbp = numeric(d, "Children in CBP custody", "cbp_care", "CBP Custody", "cbp_custody")
    hhs = numeric(d, "Children in HHS Care", "hhs_care", "HHS Care", "hhs load")
    transfers = numeric(d, "Children transferred out of CBP custody", "transfers", "Transfers")
    discharges = numeric(d, "Children discharged from HHS Care", "discharges", "Discharges")
    net = numeric(d, "Net Intake Pressure", "net_intake_pressure", "Net Intake")
    growth = numeric(d, "Daily Load Growth %", "load_growth_pct", "Load Growth")
    vol = numeric(d, "Load Volatility 7D", "load_roll_std_7", "Load Volatility")

    # If exact total load feature is absent, total system load is CBP + HHS.
    if d["Total Load"].isna().all():
        d["Total Load"] = cbp + hhs

    d["CBP Custody"] = cbp
    d["HHS Care"] = hhs
    d["Transfers"] = transfers
    d["Discharges"] = discharges
    d["Net Intake"] = net
    d["Load Growth"] = growth
    d["Load Volatility"] = vol

    return d


@st.cache_data(show_spinner=False)
def load_outputs():
    main = read_csv(DATA_DIR, "uac_features.csv", "uac_cleaned.csv", dates=True)
    anomaly = read_csv(DATA_DIR, "uac_anomaly_results.csv", "anomaly_results.csv", dates=True)
    cluster = read_csv(DATA_DIR, "uac_operational_clusters.csv", "cluster_results.csv", dates=True)
    stress = read_csv(DATA_DIR, "uac_stress_labels.csv", dates=True)

    load_results = read_csv(REPORT_DIR, "load_model_results.csv", "regression_model_results.csv")
    predictions = read_csv(REPORT_DIR, "load_predictions.csv")
    stress_results = read_csv(REPORT_DIR, "stress_model_results.csv")
    sarima = read_csv(REPORT_DIR, "sarima_forecast.csv", "forecast_30_day.csv", dates=True)
    forecast_results = read_csv(REPORT_DIR, "forecast_results.csv")
    shap = read_csv(REPORT_DIR, "shap_feature_importance.csv", "shap_global_importance.csv")
    validation = read_csv(REPORT_DIR, "final_ml_validation_report.csv")

    return main, anomaly, cluster, stress, load_results, predictions, stress_results, sarima, forecast_results, shap, validation


(
    raw_main,
    anomaly_df,
    cluster_df,
    stress_df,
    load_results,
    predictions,
    stress_results,
    sarima_df,
    forecast_results,
    shap_df,
    validation_df,
) = load_outputs()

if raw_main.empty:
    found_files = []
    if DATA_DIR.exists():
        found_files = sorted(
            p.name for p in DATA_DIR.iterdir() if p.is_file()
        )

    st.error(
        "UAC dataset could not be loaded.\n\n"
        f"Project root: `{ROOT}`\n\n"
        f"Data directory: `{DATA_DIR}`\n\n"
        f"Data directory exists: `{DATA_DIR.exists()}`\n\n"
        f"Files found in the data directory: `{found_files}`\n\n"
        "Expected one of: `uac_features.csv` or `uac_cleaned.csv`."
    )
    st.stop()

main = canonicalize_main(raw_main)

# Merge saved outputs by date. Never overwrite core engineered values.
for extra in (stress_df, anomaly_df, cluster_df):
    if not extra.empty and "Date" in extra.columns:
        cols = [c for c in extra.columns if c != "Date"]
        for c in cols:
            if c in main.columns:
                continue
            temp = extra[["Date", c]].copy()
            main = main.merge(temp, on="Date", how="left")

# Exact saved stress output drives risk when available.
stress_col = find_col(main, "Stress Level", "stress_level")
stress_score_col = find_col(main, "Stress Score", "stress_score")
if stress_col:
    main["Risk"] = main[stress_col].astype(str).replace({"nan": "Unknown"})
else:
    main["Risk"] = "Unknown"
main["Stress Score"] = numeric(main, "Stress Score", "stress_score") if stress_score_col else np.nan

cluster_col = find_col(main, "Operational Cluster", "cluster", "Cluster")
pattern_col = find_col(main, "Operational Pattern", "pattern", "Operational Pattern")
main["Cluster"] = main[cluster_col] if cluster_col else "—"
main["Pattern"] = main[pattern_col].astype(str) if pattern_col else "Operational pattern"

anomaly_flag_col = find_col(main, "Anomaly Flag", "anomaly_flag")
anomaly_status_col = find_col(main, "Anomaly Status", "anomaly_status")
anomaly_score_col = find_col(main, "Anomaly Score", "anomaly_score")
main["Anomaly"] = pd.to_numeric(main[anomaly_flag_col], errors="coerce").fillna(0) if anomaly_flag_col else 0
main["Anomaly Status"] = main[anomaly_status_col].astype(str) if anomaly_status_col else "Normal"
main["Anomaly Score"] = pd.to_numeric(main[anomaly_score_col], errors="coerce") if anomaly_score_col else np.nan

# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------
PAGES = [
    ("⌂", "Overview"),
    ("▥", "Capacity & Load"),
    ("↗", "Forecasting"),
    ("⚠", "Risk & Anomalies"),
    ("◈", "Operational Patterns"),
    ("✦", "Explainable AI"),
    ("▤", "Model Performance"),
    ("⇩", "Reports & Download"),
]
if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown("<div class='nav-title'>👥 UAC-CareAI</div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-sub'>System Capacity & Care-Load<br>Analytics Platform</div>", unsafe_allow_html=True)

    for icon, label in PAGES:
        active = st.session_state.page == label
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.page = label
            st.rerun()

    st.divider()
    st.markdown("<div class='section-label'>Data status</div>", unsafe_allow_html=True)
    st.write(f"🟢 {len(main):,} reporting observations")
    st.write(f"🟢 Latest data: {main.Date.max():%d %b %Y}")
    st.write("🟢 ML outputs loaded")
    st.write("🟢 Forecast output loaded" if not sarima_df.empty else "🟡 Forecast output missing")

    st.divider()
    st.markdown("<div class='section-label'>Appearance</div>", unsafe_allow_html=True)
    toggle = st.toggle("Dark mode", value=st.session_state.uac_dark)
    if toggle != st.session_state.uac_dark:
        st.session_state.uac_dark = toggle
        st.rerun()

    st.divider()
    st.caption("UAC-CareAI v1.0")
    st.caption("ML • Forecasting • Risk • Explainability")

# ------------------------------------------------------------
# Executive header + global filters
# ------------------------------------------------------------
page = st.session_state.page
min_date, max_date = main.Date.min().date(), main.Date.max().date()

st.markdown(
    f"""
    <div class='top-header'>
        <div>
            <div class='top-brand'>UAC-CareAI</div>
            <div class='top-sub'>System Capacity &amp; Care-Load Analytics Platform</div>
        </div>
        <div class='top-right'>
            <div>Last Updated<br><b style='color:{TEXT}'>{max_date:%d %b %Y}</b></div>
            <div class='live-badge'><span class='status-dot'></span>Live data</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Clear, labeled controls in a dedicated panel.
with st.container(border=True):
    st.markdown("<div class='filter-title'>Dashboard Controls</div>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns([1.15, 1.15, .9, 1.05, .8])
    with f1:
        st.markdown("<div class='filter-label'>Date from</div>", unsafe_allow_html=True)
        start = st.date_input("Date from", min_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
        st.markdown("<div class='filter-caption'>Start of selected period</div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div class='filter-label'>Date to</div>", unsafe_allow_html=True)
        end = st.date_input("Date to", max_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
        st.markdown("<div class='filter-caption'>End of selected period</div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div class='filter-label'>Granularity</div>", unsafe_allow_html=True)
        granularity = st.selectbox("Granularity", ["Daily", "Weekly", "Monthly"], label_visibility="collapsed")
        st.markdown("<div class='filter-caption'>Chart aggregation</div>", unsafe_allow_html=True)
    with f4:
        available_risks = sorted([x for x in main["Risk"].dropna().astype(str).unique() if x.lower() != "unknown"])
        risk_options = ["All"] + available_risks
        st.markdown("<div class='filter-label'>Risk level</div>", unsafe_allow_html=True)
        risk_filter = st.selectbox("Risk level", risk_options, label_visibility="collapsed")
        st.markdown("<div class='filter-caption'>Saved stress classification</div>", unsafe_allow_html=True)
    with f5:
        st.markdown("<div class='filter-label'>Data status</div>", unsafe_allow_html=True)
        st.markdown("<div class='filter-status'><span></span>Live data loaded</div>", unsafe_allow_html=True)
        st.markdown("<div class='filter-caption'>Existing project outputs</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

if pd.Timestamp(start) > pd.Timestamp(end):
    st.error("The start date must be earlier than the end date.")
    st.stop()

base = main[main.Date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
if risk_filter != "All":
    base = base[base["Risk"].astype(str).str.casefold() == risk_filter.casefold()]

if base.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------
def template():
    return "plotly_dark" if DARK else "plotly_white"


def style_fig(fig, height=340, legend=True):
    fig.update_layout(
        template=template(),
        height=height,
        margin=dict(l=16, r=16, t=42, b=24),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, size=11),
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT),
        legend=dict(orientation="h", y=-0.18) if legend else dict(),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


def header(title, subtitle, kicker):
    st.markdown(f"<div class='kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def delta_pct(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2 or s.iloc[-2] == 0:
        return None
    return (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100


def aggregate(data):
    d = data.copy().set_index("Date")
    agg = {
        "Total Load": "mean",
        "CBP Custody": "mean",
        "HHS Care": "mean",
        "Transfers": "sum",
        "Discharges": "sum",
        "Net Intake": "mean",
    }
    if granularity == "Weekly":
        return d.resample("W").agg(agg).reset_index()
    if granularity == "Monthly":
        return d.resample("MS").agg(agg).reset_index()
    return data.copy()


def risk_color(level):
    s = str(level).lower()
    if "critical" in s or "high" in s:
        return RED
    if "moderate" in s or "elevated" in s:
        return ORANGE
    if "low" in s or "normal" in s:
        return GREEN
    return MUTED


def latest_row(data):
    return data.sort_values("Date").iloc[-1]


def csv_bytes(data):
    return data.to_csv(index=False).encode("utf-8")


def saved_file_status():
    outputs = [
        ("uac_features.csv", DATA_DIR / "uac_features.csv"),
        ("uac_anomaly_results.csv", DATA_DIR / "uac_anomaly_results.csv"),
        ("uac_operational_clusters.csv", DATA_DIR / "uac_operational_clusters.csv"),
        ("uac_stress_labels.csv", DATA_DIR / "uac_stress_labels.csv"),
        ("load_model_results.csv", REPORT_DIR / "load_model_results.csv"),
        ("load_predictions.csv", REPORT_DIR / "load_predictions.csv"),
        ("stress_model_results.csv", REPORT_DIR / "stress_model_results.csv"),
        ("sarima_forecast.csv", REPORT_DIR / "sarima_forecast.csv"),
        ("forecast_results.csv", REPORT_DIR / "forecast_results.csv"),
        ("shap_feature_importance.csv", REPORT_DIR / "shap_feature_importance.csv"),
        ("final_ml_validation_report.csv", REPORT_DIR / "final_ml_validation_report.csv"),
    ]
    return pd.DataFrame([{"Artifact": n, "Status": "Available" if p.exists() else "Missing"} for n, p in outputs])


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    header("Overview", "Executive view of system capacity, care load, operational flows and saved risk signals.", "EXECUTIVE COMMAND CENTER")

    latest = latest_row(base)
    status = str(latest["Risk"])
    status_color = risk_color(status)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Current System Load", f"{latest['Total Load']:,.0f}", f"{delta_pct(base['Total Load']):+.1f}%" if delta_pct(base["Total Load"]) is not None else None)
    k2.metric("CBP Custody", f"{latest['CBP Custody']:,.0f}", f"{delta_pct(base['CBP Custody']):+.1f}%" if delta_pct(base["CBP Custody"]) is not None else None)
    k3.metric("HHS Care", f"{latest['HHS Care']:,.0f}", f"{delta_pct(base['HHS Care']):+.1f}%" if delta_pct(base["HHS Care"]) is not None else None)
    k4.metric("Net Intake Pressure", f"{latest['Net Intake']:+,.0f}", f"{delta_pct(base['Net Intake']):+.1f}%" if delta_pct(base["Net Intake"]) is not None else None)
    k5.metric("Saved Risk Level", status, "From stress output" if status != "Unknown" else "Not available")

    chart = aggregate(base)

    a, b = st.columns([1.65, 1])
    with a:
        with st.container(border=True):
            st.markdown("### 📈 System Load Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart.Date, y=chart["Total Load"], name="Total System Load", mode="lines", line=dict(color=BLUE, width=3)))
            fig.add_trace(go.Scatter(x=chart.Date, y=chart["CBP Custody"], name="CBP Custody", mode="lines", line=dict(color=PURPLE, width=1.8)))
            fig.add_trace(go.Scatter(x=chart.Date, y=chart["HHS Care"], name="HHS Care", mode="lines", line=dict(color=TEAL, width=1.8)))
            fig.update_layout(hovermode="x unified")
            style_fig(fig, 405)
            st.plotly_chart(fig, use_container_width=True, key="ov_load")

    with b:
        with st.container(border=True):
            st.markdown("### 🚨 Recent Alerts")
            recent = base.sort_values("Date", ascending=False)
            rows = []
            for _, r in recent.iterrows():
                if r["Anomaly"] == 1:
                    rows.append(("Anomaly", r["Date"], RED, str(r["Anomaly Status"])))
                elif str(r["Risk"]).lower() in {"high", "critical"}:
                    rows.append(("High stress level", r["Date"], risk_color(r["Risk"]), str(r["Risk"])))
                if len(rows) >= 6:
                    break
            if not rows:
                st.success("No saved high-priority alert condition in this filtered period.")
            for title, dt, color, detail in rows:
                st.markdown(
                    f"<div class='alert-row'><b style='color:{color}'>●</b> {title}<br><span class='small'>{dt:%d %b %Y} · {detail}</span></div>",
                    unsafe_allow_html=True,
                )

    a, b, c = st.columns([1, 1, 1.15])
    with a:
        with st.container(border=True):
            st.markdown("### Risk / Stress Distribution")
            r = base["Risk"].value_counts().rename_axis("Risk").reset_index(name="Observations")
            if not r.empty:
                fig = px.pie(r, names="Risk", values="Observations", hole=.62, color="Risk",
                             color_discrete_map={x: risk_color(x) for x in r["Risk"]})
                fig.update_traces(textinfo="percent", textposition="inside")
                style_fig(fig, 330)
                st.plotly_chart(fig, use_container_width=True, key="ov_risk")

    with b:
        with st.container(border=True):
            st.markdown("### Transfers vs Discharges")
            flow = aggregate(base)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=flow.Date, y=flow["Transfers"], name="Transfers", marker_color=BLUE))
            fig.add_trace(go.Bar(x=flow.Date, y=flow["Discharges"], name="Discharges", marker_color=TEAL))
            fig.update_layout(barmode="group")
            style_fig(fig, 330)
            st.plotly_chart(fig, use_container_width=True, key="ov_flow")

    with c:
        with st.container(border=True):
            st.markdown("### Key Takeaways")
            d = delta_pct(base["Total Load"])
            takeaways = [
                f"Latest system load is <b>{latest['Total Load']:,.0f}</b> children.",
                f"Saved stress level is <b>{status}</b>." if status != "Unknown" else "Saved stress level is not available for the current observation.",
                f"Selected-period anomaly observations: <b>{int(base['Anomaly'].sum()):,}</b>.",
                f"Latest net intake pressure is <b>{latest['Net Intake']:+,.0f}</b>.",
            ]
            if d is not None:
                takeaways.insert(1, f"Latest load changed <b>{abs(d):.1f}%</b> versus the prior observation.")
            for item in takeaways:
                st.markdown(f"<div class='insight-row'>• {item}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 🔮 Forecast Snapshot")
        if sarima_df.empty:
            st.info("No saved SARIMA output found.")
        else:
            f = sarima_df.copy()
            fc = find_col(f, "Forecast Load", "Forecast", "forecast")
            actual = find_col(f, "Actual Load", "Actual", "actual")
            if fc:
                f[fc] = numeric(f, fc)
                f = f.dropna(subset=[fc]).sort_values("Date")
                if actual:
                    f[actual] = numeric(f, actual)
                c1, c2, c3 = st.columns(3)
                c1.metric("Saved Forecast Rows", f"{len(f):,}")
                c2.metric("Latest Saved Forecast", f"{f.iloc[-1][fc]:,.0f}")
                c3.metric("Peak Saved Forecast", f"{f[fc].max():,.0f}")
                fig = go.Figure()
                if actual:
                    fig.add_trace(go.Scatter(x=f.Date, y=f[actual], name="Actual", line=dict(color=BLUE, width=2)))
                fig.add_trace(go.Scatter(x=f.Date, y=f[fc], name="SARIMA", line=dict(color=PURPLE, width=2.5, dash="dash")))
                style_fig(fig, 300)
                st.plotly_chart(fig, use_container_width=True, key="ov_forecast")
                st.caption("The dashboard labels the saved file as forecast output; it does not assume validation rows are future projections.")

# ============================================================
# CAPACITY & LOAD
# ============================================================
elif page == "Capacity & Load":
    header("Capacity & Load", "Interactive historical view of system load, custody, care and operational flows.", "CAPACITY ANALYTICS")

    latest = latest_row(base)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average System Load", f"{base['Total Load'].mean():,.0f}")
    c2.metric("Peak System Load", f"{base['Total Load'].max():,.0f}")
    c3.metric("Transfers in Selection", f"{base['Transfers'].sum():,.0f}")
    c4.metric("Discharges in Selection", f"{base['Discharges'].sum():,.0f}")

    chart = aggregate(base)

    a, b = st.columns([1.65, 1])
    with a:
        with st.container(border=True):
            st.markdown("### Total System Load Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart.Date, y=chart["Total Load"], name="Total Load", line=dict(color=BLUE, width=3)))
            fig.add_trace(go.Scatter(x=chart.Date, y=chart["CBP Custody"], name="CBP Custody", line=dict(color=PURPLE, width=2)))
            fig.add_trace(go.Scatter(x=chart.Date, y=chart["HHS Care"], name="HHS Care", line=dict(color=TEAL, width=2)))
            fig.update_layout(hovermode="x unified")
            style_fig(fig, 430)
            st.plotly_chart(fig, use_container_width=True, key="cap_load")

    with b:
        with st.container(border=True):
            st.markdown("### Current Load Composition")
            vals = pd.DataFrame({"Component": ["CBP Custody", "HHS Care"], "Value": [latest["CBP Custody"], latest["HHS Care"]]})
            fig = px.pie(vals, names="Component", values="Value", hole=.62,
                         color="Component", color_discrete_map={"CBP Custody": PURPLE, "HHS Care": TEAL})
            fig.update_traces(textinfo="label+percent")
            style_fig(fig, 430)
            st.plotly_chart(fig, use_container_width=True, key="cap_donut")

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("### Transfers vs Discharges")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=chart.Date, y=chart["Transfers"], name="Transfers", marker_color=BLUE))
            fig.add_trace(go.Bar(x=chart.Date, y=chart["Discharges"], name="Discharges", marker_color=TEAL))
            fig.update_layout(barmode="group")
            style_fig(fig, 360)
            st.plotly_chart(fig, use_container_width=True, key="cap_flow")

    with b:
        with st.container(border=True):
            st.markdown("### Net Intake Pressure")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=chart.Date, y=chart["Net Intake"], name="Net Intake Pressure", marker_color=ORANGE))
            style_fig(fig, 360)
            st.plotly_chart(fig, use_container_width=True, key="cap_net")

    with st.container(border=True):
        st.markdown("### 7-Day / 14-Day Load Trend")
        roll = base[["Date", "Total Load"]].sort_values("Date").copy()
        roll["7-Day Average"] = roll["Total Load"].rolling(7, min_periods=1).mean()
        roll["14-Day Average"] = roll["Total Load"].rolling(14, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll.Date, y=roll["Total Load"], name="Total Load", line=dict(color=MUTED, width=1)))
        fig.add_trace(go.Scatter(x=roll.Date, y=roll["7-Day Average"], name="7-Day Average", line=dict(color=BLUE, width=2.5)))
        fig.add_trace(go.Scatter(x=roll.Date, y=roll["14-Day Average"], name="14-Day Average", line=dict(color=RED, width=2)))
        style_fig(fig, 370)
        st.plotly_chart(fig, use_container_width=True, key="cap_roll")

# ============================================================
# FORECASTING
# ============================================================
elif page == "Forecasting":
    header("Forecasting", "Forward-looking analysis from the saved SARIMA/SARIMAX output. No forecast is invented in the dashboard.", "PREDICTIVE ANALYTICS")

    if sarima_df.empty:
        st.warning("Saved SARIMA forecast output was not found.")
    else:
        f = sarima_df.copy().dropna(subset=["Date"]).sort_values("Date")
        fc = find_col(f, "Forecast Load", "Forecast", "forecast")
        actual = find_col(f, "Actual Load", "Actual", "actual")
        err = find_col(f, "Forecast Error", "Error", "error")

        if not fc:
            st.error("Forecast file exists, but no forecast column was found.")
        else:
            f[fc] = numeric(f, fc)
            if actual:
                f[actual] = numeric(f, actual)
            if err:
                f[err] = numeric(f, err)
            f = f.dropna(subset=[fc])

            has_validation_actual = actual is not None and f[actual].notna().any()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Saved Forecast Rows", f"{len(f):,}")
            c2.metric("Mean Forecast", f"{f[fc].mean():,.0f}")
            c3.metric("Peak Forecast", f"{f[fc].max():,.0f}")
            c4.metric("Output Type", "Validation output" if has_validation_actual else "Future output")

            with st.container(border=True):
                st.markdown("### SARIMA Forecast Output")
                fig = go.Figure()
                if actual:
                    fig.add_trace(go.Scatter(x=f.Date, y=f[actual], name="Actual Load", line=dict(color=BLUE, width=2)))
                fig.add_trace(go.Scatter(x=f.Date, y=f[fc], name="Forecast Load", line=dict(color=PURPLE, width=2.5, dash="dash")))
                lower = find_col(f, "Lower 95", "Lower_95", "Lower")
                upper = find_col(f, "Upper 95", "Upper_95", "Upper")
                if lower and upper:
                    f[lower] = numeric(f, lower)
                    f[upper] = numeric(f, upper)
                    fig.add_trace(go.Scatter(x=f.Date, y=f[upper], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=f.Date, y=f[lower], fill="tonexty", line=dict(width=0), name="95% interval", hoverinfo="skip"))
                style_fig(fig, 455)
                st.plotly_chart(fig, use_container_width=True, key="fc_main")

            if err and f[err].notna().any():
                e = f.dropna(subset=[err])
                mae = e[err].abs().mean()
                rmse = np.sqrt(np.mean(e[err] ** 2))
                a, b, c = st.columns(3)
                a.metric("Mean Absolute Error", f"{mae:,.2f}")
                b.metric("RMSE", f"{rmse:,.2f}")
                c.metric("Mean Error", f"{e[err].mean():,.2f}")
                with st.container(border=True):
                    st.markdown("### Forecast Error")
                    fig = px.bar(e, x="Date", y=err, color_discrete_sequence=[ORANGE])
                    style_fig(fig, 330, legend=False)
                    st.plotly_chart(fig, use_container_width=True, key="fc_error")

            with st.container(border=True):
                st.markdown("### Saved Forecast Records")
                st.dataframe(f.tail(40).round(3), use_container_width=True, hide_index=True)

# ============================================================
# RISK & ANOMALIES
# ============================================================
elif page == "Risk & Anomalies":
    header("Risk & Anomalies", "Review saved stress labels and anomaly-detection outputs for unusual operating conditions.", "RISK INTELLIGENCE")

    latest = latest_row(base)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Saved Risk", str(latest["Risk"]))
    c2.metric("Latest Stress Score", f"{latest['Stress Score']:.2f}" if pd.notna(latest["Stress Score"]) else "N/A")
    c3.metric("Anomaly Observations", f"{int(base['Anomaly'].sum()):,}")
    c4.metric("Latest Volatility", f"{latest['Load Volatility']:,.1f}" if pd.notna(latest["Load Volatility"]) else "N/A")

    a, b = st.columns([1.5, 1])
    with a:
        with st.container(border=True):
            st.markdown("### 🚨 Anomaly Timeline")
            ad = base[base["Anomaly"] == 1].copy()
            if ad.empty:
                st.success("No anomaly observations in the selected period.")
            else:
                fig = px.scatter(
                    ad, x="Date", y="Total Load", size="Anomaly Score",
                    color="Risk", hover_data=["Net Intake", "Load Growth", "Anomaly Status"],
                    color_discrete_map={x: risk_color(x) for x in ad["Risk"].astype(str).unique()},
                )
                style_fig(fig, 420)
                st.plotly_chart(fig, use_container_width=True, key="risk_anomaly")

    with b:
        with st.container(border=True):
            st.markdown("### Stress Level Distribution")
            r = base["Risk"].value_counts().rename_axis("Risk").reset_index(name="Observations")
            fig = px.bar(r, x="Risk", y="Observations", color="Risk",
                         color_discrete_map={x: risk_color(x) for x in r["Risk"]})
            fig.update_layout(showlegend=False)
            style_fig(fig, 420, legend=False)
            st.plotly_chart(fig, use_container_width=True, key="risk_dist")

    with st.container(border=True):
        st.markdown("### Saved Anomaly / Risk Records")
        cols = ["Date", "Total Load", "Net Intake", "Risk", "Stress Score", "Anomaly", "Anomaly Status", "Anomaly Score"]
        cols = [c for c in cols if c in base.columns]
        st.dataframe(base.sort_values("Date", ascending=False)[cols].head(100).round(3), use_container_width=True, hide_index=True)

# ============================================================
# OPERATIONAL PATTERNS
# ============================================================
elif page == "Operational Patterns":
    header("Operational Patterns", "Explore the saved K-Means operational clusters and their observed load conditions.", "OPERATIONAL INTELLIGENCE")

    if cluster_df.empty and main["Cluster"].eq("—").all():
        st.warning("Operational clustering output was not found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters Observed", f"{main['Cluster'].nunique(dropna=True):,}")
        c2.metric("Largest Cluster", str(main["Cluster"].value_counts().index[0]) if not main["Cluster"].empty else "—")
        c3.metric("Selected Observations", f"{len(base):,}")

        a, b = st.columns([1, 1.4])
        with a:
            with st.container(border=True):
                st.markdown("### Cluster Distribution")
                cc = base["Cluster"].astype(str).value_counts().rename_axis("Cluster").reset_index(name="Observations")
                fig = px.bar(cc, x="Cluster", y="Observations", color="Cluster")
                fig.update_layout(showlegend=False)
                style_fig(fig, 360, legend=False)
                st.plotly_chart(fig, use_container_width=True, key="cluster_dist")

        with b:
            with st.container(border=True):
                st.markdown("### Load vs Intake by Cluster")
                cd = base.copy()
                fig = px.scatter(cd, x="Net Intake", y="Total Load", color="Cluster",
                                 hover_data=["Date", "CBP Custody", "HHS Care", "Pattern"])
                style_fig(fig, 360)
                st.plotly_chart(fig, use_container_width=True, key="cluster_scatter")

        with st.container(border=True):
            st.markdown("### Operational Pattern Summary")
            summary = (
                base.groupby(["Cluster", "Pattern"], dropna=False)
                .agg(
                    Observations=("Date", "count"),
                    Avg_Load=("Total Load", "mean"),
                    Avg_Net_Intake=("Net Intake", "mean"),
                    Avg_CBP=("CBP Custody", "mean"),
                    Avg_HHS=("HHS Care", "mean"),
                )
                .reset_index()
                .sort_values("Observations", ascending=False)
            )
            st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

# ============================================================
# EXPLAINABLE AI
# ============================================================
elif page == "Explainable AI":
    header("Explainable AI", "Global SHAP feature influence for the saved load-prediction model.", "MODEL EXPLAINABILITY")

    if shap_df.empty:
        st.warning("SHAP feature-importance output was not found.")
    else:
        feature = find_col(shap_df, "Feature", "feature")
        importance = find_col(shap_df, "Mean Absolute SHAP", "Mean_Absolute_SHAP", "mean_absolute_shap", "importance")
        if not feature or not importance:
            st.error("The SHAP report does not contain the expected feature and importance columns.")
        else:
            s = shap_df.copy()
            s[importance] = numeric(s, importance)
            s = s.dropna(subset=[importance]).sort_values(importance, ascending=False)
            top = s.head(12).sort_values(importance)

            a, b = st.columns([1.5, .75])
            with a:
                with st.container(border=True):
                    st.markdown("### Global Feature Influence")
                    fig = px.bar(top, x=importance, y=feature, orientation="h")
                    fig.update_traces(marker_color=PURPLE)
                    style_fig(fig, 470, legend=False)
                    st.plotly_chart(fig, use_container_width=True, key="shap")
            with b:
                with st.container(border=True):
                    st.markdown("### Explanation")
                    st.metric("Top Feature", str(s.iloc[0][feature]))
                    st.metric("Features Analysed", f"{len(s):,}")
                    st.write("Mean absolute SHAP measures the average magnitude of each feature's contribution to the saved model predictions.")
                    st.caption("This page explains the model; it does not retrain or alter it.")

            with st.container(border=True):
                st.markdown("### SHAP Importance Table")
                st.dataframe(s.head(25).round(5), use_container_width=True, hide_index=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    header("Model Performance", "Evaluate the saved regression and stress-classification results produced by the ML pipeline.", "MODEL GOVERNANCE")

    if not load_results.empty:
        model = find_col(load_results, "Model")
        rmse = find_col(load_results, "RMSE")
        mae = find_col(load_results, "MAE")
        r2 = find_col(load_results, "R2", "R²")

        if model and rmse:
            lr = load_results.copy()
            for c in [rmse, mae, r2]:
                if c:
                    lr[c] = numeric(lr, c)
            best = lr.dropna(subset=[rmse]).sort_values(rmse).iloc[0]

            a, b, c, d = st.columns(4)
            a.metric("Best Saved Model", str(best[model]))
            b.metric("RMSE", f"{best[rmse]:.2f}")
            c.metric("MAE", f"{best[mae]:.2f}" if mae else "N/A")
            d.metric("R²", f"{best[r2]:.3f}" if r2 else "N/A")

            x, y = st.columns(2)
            with x:
                with st.container(border=True):
                    fig = px.bar(lr.sort_values(rmse), x=model, y=rmse, text_auto=".2f")
                    fig.update_traces(marker_color=BLUE)
                    style_fig(fig, 350, legend=False)
                    fig.update_layout(title="RMSE — Lower is Better")
                    st.plotly_chart(fig, use_container_width=True, key="perf_rmse")
            with y:
                with st.container(border=True):
                    if r2:
                        fig = px.bar(lr.sort_values(r2, ascending=False), x=model, y=r2, text_auto=".3f")
                        fig.update_traces(marker_color=TEAL)
                        style_fig(fig, 350, legend=False)
                        fig.update_layout(title="R² — Higher is Better")
                        st.plotly_chart(fig, use_container_width=True, key="perf_r2")

            with st.container(border=True):
                st.markdown("### Regression Benchmark")
                st.dataframe(lr.round(4), use_container_width=True, hide_index=True)

    if not predictions.empty:
        date_c = find_col(predictions, "Date")
        actual = find_col(predictions, "Actual Load", "Actual", "y_true")
        pred = find_col(predictions, "Predicted Load", "Predicted", "Prediction", "y_pred")
        if actual and pred:
            p = predictions.copy()
            xvals = pd.to_datetime(p[date_c], errors="coerce") if date_c else np.arange(len(p))
            p[actual] = numeric(p, actual)
            p[pred] = numeric(p, pred)
            with st.container(border=True):
                st.markdown("### Actual vs Predicted Load")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xvals, y=p[actual], name="Actual", line=dict(color=BLUE, width=2.2)))
                fig.add_trace(go.Scatter(x=xvals, y=p[pred], name="Predicted", line=dict(color=PURPLE, width=2, dash="dash")))
                style_fig(fig, 400)
                st.plotly_chart(fig, use_container_width=True, key="perf_pred")

    if not stress_results.empty:
        model = find_col(stress_results, "Model")
        f1 = find_col(stress_results, "F1", "F1 Score", "f1_score")
        if model and f1:
            sr = stress_results.copy()
            sr[f1] = numeric(sr, f1)
            with st.container(border=True):
                st.markdown("### Stress Classification Benchmark")
                fig = px.bar(sr, x=model, y=f1, text_auto=".3f")
                fig.update_traces(marker_color=PURPLE)
                style_fig(fig, 350, legend=False)
                st.plotly_chart(fig, use_container_width=True, key="perf_f1")
                st.dataframe(sr.round(4), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("### Saved AI Component Status")
        status = [
            ["Load Prediction", "Regression benchmark", "Available" if not load_results.empty else "Missing"],
            ["Stress Classification", "Classification benchmark", "Available" if not stress_results.empty else "Missing"],
            ["Anomaly Detection", "Isolation Forest output", "Available" if not anomaly_df.empty else "Missing"],
            ["Operational Clustering", "K-Means output", "Available" if not cluster_df.empty else "Missing"],
            ["Forecasting", "SARIMA/SARIMAX output", "Available" if not sarima_df.empty else "Missing"],
            ["Explainability", "SHAP output", "Available" if not shap_df.empty else "Missing"],
        ]
        st.dataframe(pd.DataFrame(status, columns=["Component", "Evidence", "Status"]), use_container_width=True, hide_index=True)

# ============================================================
# REPORTS
# ============================================================
elif page == "Reports & Download":
    header("Reports & Download", "Export the filtered evidence used by the dashboard and inspect artifact availability.", "PROJECT EVIDENCE")

    c1, c2, c3 = st.columns(3)
    c1.metric("Source Observations", f"{len(main):,}")
    c2.metric("Full Date Coverage", f"{main.Date.min():%d %b %Y} → {main.Date.max():%d %b %Y}")
    c3.metric("Filtered Observations", f"{len(base):,}")

    with st.container(border=True):
        st.markdown("### Export Current Selection")
        st.download_button(
            "⬇ Download filtered CSV",
            data=csv_bytes(base),
            file_name="uac_careai_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown("### Analytical Artifact Status")
        st.dataframe(saved_file_status(), use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("### Filtered Data Preview")
        st.dataframe(base.sort_values("Date", ascending=False).head(75).round(3), use_container_width=True, hide_index=True)

st.markdown(
    "<div class='footer'>UAC-CareAI • System Capacity & Care-Load Analytics • Existing ML outputs only</div>",
    unsafe_allow_html=True,
)
