import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# ─────────────────────────────────────────
# CONFIGURAZIONE PAGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RRG Professional Analyzer",
    page_icon="🔄",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #0A1628 0%, #1C3057 100%);
        padding: 24px; border-radius: 16px; margin-bottom: 24px; color: white;
    }
    
    .q-card { padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .q-leading { background-color: rgba(0,135,90,0.1); border-color: #00875A; color: #00875A; }
    .q-weakening { background-color: rgba(196,121,0,0.1); border-color: #C47900; color: #C47900; }
    .q-lagging { background-color: rgba(196,0,43,0.1); border-color: #C4002B; color: #C4002B; }
    .q-improving { background-color: rgba(0,98,196,0.1); border-color: #0062C4; color: #0062C4; }
    
    .asset-name { font-weight: bold; color: #0A1628; font-size: 15px; margin-bottom: 2px; }
    .quad-label { font-weight: 800; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
    .trend-text { font-size: 11px; color: #4A5A7A; margin-top: 6px; line-height: 1.4; font-style: italic; }
    .metrics-text { font-family: 'DM Mono'; font-size: 10px; color: #4A5A7A; margin-top: 4px; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────

def compute_rrg(df, benchmark_col, sector_cols, ema_short=12, ema_long=26, z_window=52, m_window=14):
    results = {}
    for col in sector_cols:
        rs_raw = df[col] / df[benchmark_col]
        rs_ema_short = rs_raw.ewm(span=ema_short, adjust=False).mean()
        rs_smoothed = rs_ema_short.ewm(span=ema_long, adjust=False).mean()
        
        roll_mean = rs_smoothed.rolling(window=z_window).mean()
        roll_std = rs_smoothed.rolling(window=z_window).std(ddof=0)
        rs_ratio = 100 + ((rs_smoothed - roll_mean) / roll_std.replace(0, np.nan)) * 10
        
        rs_ratio_diff = rs_ratio.diff()
        m_mean = rs_ratio_diff.rolling(window=m_window).mean()
        m_std = rs_ratio_diff.rolling(window=m_window).std(ddof=0)
        rs_mom = 100 + ((rs_ratio_diff - m_mean) / m_std.replace(0, np.nan)) * 10
        
        results[col] = {"rs_ratio": rs_ratio, "rs_momentum": rs_mom}
    return results

def get_quadrant_info(x, y):
    if x >= 100 and y >= 100: return "Leading", "q-leading"
    if x >= 100 and y < 100: return "Weakening", "q-weakening"
    if x < 100 and y < 100: return "Lagging", "q-lagging"
    return "Improving", "q-improving"

def analyze_trend(name, rx_series, rm_series, lookback=5):
    """Analisi del movimento degli ultimi periodi."""
    if len(rx_series) < lookback + 1: return "Dati storici insufficienti per trend."
    
    curr_x, curr_y = rx_series.iloc[-1], rm_series.iloc[-1]
    prev_x, prev_y = rx_series.iloc[-lookback], rm_series.iloc[-lookback]
    
    curr_q, _ = get_quadrant_info(curr_x, curr_y)
    prev_q, _ = get_quadrant_info(prev_x, prev_y)
    
    if curr_q == prev_q:
        direction_x = "rinforzando" if curr_x > prev_x else "indebolendo"
        return f"Stazionario in {curr_q}, si sta {direction_x} ulteriormente."
    else:
        return f"Passato da {prev_q} a {curr_q}. Il momentum è {'in crescita' if curr_y > prev_y else 'in calo'}."

# ─────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────

def parse_file(uploaded):
    try:
        if uploaded.name.endswith('.csv'):
            sample = uploaded.read(4096).decode('utf-8', errors='ignore')
            uploaded.seek(0)
            sep = ';' if sample.count(';') > sample.count(',') else ','
            decimal = ',' if sep == ';' else '.'
            df = pd.read_csv(uploaded, sep=sep, decimal=decimal)
        else:
            df = pd.read_excel(uploaded)

        date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["date", "data", "time"])), df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        
        df = df.dropna(axis=1, how='all').ffill().dropna()
        return df
    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return None

# ─────────────────────────────────────────
# INTERFACCIA
# ─────────────────────────────────────────

st.markdown('<div class="main-header"><h1>🔄 RRG Analyzer</h1><p>Relative Rotation Graph — Metodologia JdK</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Dati")
    file = st.file_uploader("Carica File", type=['csv', 'xlsx'])
    df_raw = parse_file(file) if file else None

if df_raw is not None:
    with st.sidebar:
        bench = st.selectbox("Benchmark", df_raw.columns)
        assets = st.multiselect("Assets", [c for c in df_raw.columns if c != bench], default=[c for c in df_raw.columns if c != bench][:8])
        
        idx_min, idx_max = df_raw.index.min(), df_raw.index.max()
        if pd.isna(idx_min): st.stop()
            
        d_start = st.date_input("Inizio", value=idx_min.date())
        d_end = st.date_input("Fine", value=idx_max.date())
        
        with st.expander("Parametri Avanzati"):
            e_s, e_l = st.number_input("EMA Short", 12), st.number_input("EMA Long", 26)
            z_w, m_w = st.number_input("Z-Score Window", 52), st.number_input("Mom Window", 14)

    df_final = df_raw.loc[(df_raw.index >= pd.Timestamp(d_start)) & (df_raw.index <= pd.Timestamp(d_end))]

    if len(df_final) < z_w + m_w:
        st.warning("Dati insufficienti per i parametri selezionati.")
    else:
        results = compute_rrg(df_final, bench, assets, e_s, e_l, z_w, m_w)
        
        col_chart, col_snap = st.columns([3, 1])
        
        with col_chart:
            show_t = st.toggle("🐾 Mostra Code", value=True)
            t_len = st.slider("Lunghezza", 2, 60, 10, disabled=not show_t)
            
            fig = go.Figure()
            fig.add_hline(y=100, line_dash="dot", line_color="rgba(0,0,0,0.3)")
            fig.add_vline(x=100, line_dash="dot", line_color="rgba(0,0,0,0.3)")
            
            colors = ["#1D5FC4", "#E63946", "#2A9D8F", "#F4A261", "#8338EC", "#06D6A0", "#FB8500"]
            
            for i, (name, data) in enumerate(results.items()):
                color = colors[i % len(colors)]
                rx, rm = data["rs_ratio"].dropna(), data["rs_momentum"].dropna()
                if rx.empty: continue
                
                if show_t:
                    fig.add_trace(go.Scatter(x=rx.tail(t_len), y=rm.tail(t_len), mode='lines', line=dict(width=2, color=color), opacity=0.3, showlegend=False))
                
                fig.add_trace(go.Scatter(
                    x=[rx.iloc[-1]], y=[rm.iloc[-1]], mode='markers+text', name=name,
                    text=[name], textposition="top right",
                    marker=dict(size=14, color=color, line=dict(width=2, color='white'))
                ))

            fig.update_layout(template="plotly_white", height=650, xaxis_title="RS-RATIO (Forza Relativa)", yaxis_title="RS-MOMENTUM (Velocità)")
            st.plotly_chart(fig, use_container_width=True)

        with col_snap:
            st.subheader("📋 Analisi Trend")
            for name, data in results.items():
                rx_s, rm_s = data["rs_ratio"], data["rs_momentum"]
                val_x, val_y = rx_s.iloc[-1], rm_s.iloc[-1]
                
                q_label, q_class = get_quadrant_info(val_x, val_y)
                trend_msg = analyze_trend(name, rx_s, rm_s, lookback=5)
                
                st.markdown(f"""
                <div class="q-card {q_class}">
                    <div class="asset-name">{name}</div>
                    <div class="quad-label">{q_label}</div>
                    <div class="trend-text">{trend_msg}</div>
                    <div class="metrics-text">
                        R: <b>{val_x:.2f}</b> | M: <b>{val_y:.2f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("Carica un file per visualizzare l'analisi RRG.")
