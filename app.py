import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from datetime import datetime

# ─────────────────────────────────────────
# CONFIGURAZIONE PAGINA
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RRG Professional Analyzer",
    page_icon="🔄",
    layout="wide",
)

# CSS Custom per un look High-End
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .metric-card {
        background: white; border: 1px solid #E4EAF4; border-radius: 12px;
        padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-label { font-size: 11px; font-weight: 700; color: #8A9BBE; text-transform: uppercase; }
    .metric-value { font-size: 24px; font-weight: 700; font-family: 'DM Mono'; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CORE ENGINE (OTTIMIZZATO)
# ─────────────────────────────────────────

def compute_rrg(df, benchmark_col, sector_cols, ema_short=12, ema_long=26, z_window=52, m_window=14):
    """Calcolo RRG vettorializzato - Niente cicli for infiniti."""
    results = {}
    for col in sector_cols:
        # 1. Relative Strength Raw
        rs_raw = df[col] / df[benchmark_col]
        
        # 2. RS-Ratio (Standardizzato)
        # Usiamo ewm() di pandas: molto più veloce e stabile della funzione manuale
        rs_ema_short = rs_raw.ewm(span=ema_short, adjust=False).mean()
        rs_smoothed = rs_ema_short.ewm(span=ema_long, adjust=False).mean()
        
        roll_mean = rs_smoothed.rolling(window=z_window).mean()
        roll_std = rs_smoothed.rolling(window=z_window).std(ddof=0)
        
        # Evitiamo divisioni per zero o NaN
        rs_ratio = 100 + ((rs_smoothed - roll_mean) / roll_std.replace(0, np.nan)) * 10
        
        # 3. RS-Momentum (Rate of Change dello Z-Score)
        # Il Momentum è lo Z-Score della differenza del RS-Ratio
        rs_ratio_diff = rs_ratio.diff()
        m_mean = rs_ratio_diff.rolling(window=m_window).mean()
        m_std = rs_ratio_diff.rolling(window=m_window).std(ddof=0)
        
        rs_mom = 100 + ((rs_ratio_diff - m_mean) / m_std.replace(0, np.nan)) * 10
        
        results[col] = {"rs_ratio": rs_ratio, "rs_momentum": rs_mom}
    return results

# ─────────────────────────────────────────
# PARSING DATI (BLINDATO)
# ─────────────────────────────────────────

def parse_file(uploaded):
    try:
        if uploaded.name.endswith('.csv'):
            # Autodetect separator
            sample = uploaded.read(4096).decode('utf-8', errors='ignore')
            uploaded.seek(0)
            sep = ';' if sample.count(';') > sample.count(',') else ','
            decimal = ',' if sep == ';' else '.'
            df = pd.read_csv(uploaded, sep=sep, decimal=decimal)
        else:
            df = pd.read_excel(uploaded)

        if df.empty: raise ValueError("File vuoto.")

        # Identifica colonna date
        date_col = next((c for c in df.columns if any(k in str(c).lower() for k in ["date", "data", "time"])), df.columns[0])
        
        # Conversione date robusta
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        
        # Pulizia numerica: forza tutto a float, rimuovi sporcizia
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        
        # Rimuovi colonne completamente vuote o righe NaT
        df = df.dropna(axis=1, how='all').ffill().dropna()
        
        if df.empty: raise ValueError("Nessun dato numerico valido dopo la pulizia.")
        return df
    except Exception as e:
        st.error(f"Errore critico nel parsing: {e}")
        return None

# ─────────────────────────────────────────
# VISUALIZZAZIONE
# ─────────────────────────────────────────

def build_rrg_chart(results, show_trails, trail_length):
    fig = go.Figure()
    
    all_x, all_y = [], []
    for v in results.values():
        valid = v["rs_ratio"].dropna()
        valid_m = v["rs_momentum"].dropna()
        all_x.extend(valid.tolist())
        all_y.extend(valid_m.tolist())

    if not all_x: return fig

    # Limiti assi dinamici con buffer di sicurezza
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    pad = 2
    
    # Quadranti
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(0,0,0,0.2)")
    fig.add_vline(x=100, line_dash="dot", line_color="rgba(0,0,0,0.2)")

    # Plot dei settori
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        x_series = data["rs_ratio"].dropna()
        y_series = data["rs_momentum"].dropna()
        
        if x_series.empty: continue
        
        # Trail (Coda)
        if show_trails:
            tx = x_series.tail(trail_length)
            ty = y_series.tail(trail_length)
            fig.add_trace(go.Scatter(x=tx, y=ty, mode='lines', line=dict(width=1.5, color=color), opacity=0.4, showlegend=False))

        # Punto Corrente
        last_x, last_y = x_series.iloc[-1], y_series.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y], mode='markers+text',
            name=name, text=[name], textposition="top center",
            marker=dict(size=12, color=color, line=dict(width=1, color='white'))
        ))

    fig.update_layout(
        template="plotly_white", height=700,
        xaxis=dict(title="RS-RATIO (Forza)", range=[min(95, xmin-pad), max(105, xmax+pad)]),
        yaxis=dict(title="RS-MOMENTUM (Energia)", range=[min(95, ymin-pad), max(105, ymax+pad)]),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

st.title("🔄 RRG Professional Analyzer")

with st.sidebar:
    st.header("1. Data Ingestion")
    file = st.file_uploader("Carica Excel/CSV", type=['csv', 'xlsx'])
    
    df_raw = None
    if file:
        df_raw = parse_file(file)
        if df_raw is not None:
            st.success(f"Dati caricati: {df_raw.shape[0]} righe")

if df_raw is not None:
    with st.sidebar:
        st.header("2. Configurazione")
        bench = st.selectbox("Benchmark", df_raw.columns)
        assets = st.multiselect("Assets", [c for c in df_raw.columns if c != bench], 
                                default=[c for c in df_raw.columns if c != bench][:5])
        
        # Protezione NaT: calcoliamo i limiti solo se l'indice è valido
        idx_min = df_raw.index.min()
        idx_max = df_raw.index.max()
        
        if pd.isna(idx_min):
            st.error("Errore: Indice temporale non valido.")
            st.stop()
            
        d_start = st.date_input("Inizio", value=idx_min.date())
        d_end = st.date_input("Fine", value=idx_max.date())
        
        with st.expander("Parametri Algoritmo"):
            e_s = st.number_input("EMA Short", 12)
            e_l = st.number_input("EMA Long", 26)
            z_w = st.number_input("Z-Score Window", 52)
            m_w = st.number_input("Mom Window", 14)

    # Filtro dati
    mask = (df_raw.index >= pd.Timestamp(d_start)) & (df_raw.index <= pd.Timestamp(d_end))
    df_final = df_raw.loc[mask]

    if len(df_final) < z_w + m_w:
        st.warning(f"Dati insufficienti per i parametri scelti. Servono almeno {z_w + m_w} periodi.")
    else:
        results = compute_rrg(df_final, bench, assets, e_s, e_l, z_w, m_w)
        
        # UI Layout
        col_c, col_t = st.columns([4, 1])
        
        with col_c:
            show_t = st.checkbox("Mostra Code (Trails)", value=True)
            t_len = st.slider("Lunghezza Coda", 2, 50, 10)
            fig = build_rrg_chart(results, show_t, t_len)
            st.plotly_chart(fig, use_container_width=True)

        with col_t:
            st.subheader("Snapshot")
            for name, data in results.items():
                rx = data["rs_ratio"].iloc[-1]
                rm = data["rs_momentum"].iloc[-1]
                st.markdown(f"""
                **{name}** R: `{rx:.2f}` | M: `{rm:.2f}`
                ---
                """)
else:
    st.info("Benvenuto. Carica un file per iniziare l'analisi.")
