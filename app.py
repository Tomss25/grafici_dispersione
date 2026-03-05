import streamlit as st
import pandas as pd
import plotly.express as px

# Configurazione istituzionale: Wide layout, sidebar ridotta
st.set_page_config(page_title="Quantitative Strategy Desk", layout="wide", initial_sidebar_state="expanded")

# Iniezione CSS per layout denso stile Bloomberg Terminal
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 95%; }
    h1, h2, h3 { color: #ffffff; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #1e2530; padding: 10px; border-radius: 5px; border-left: 3px solid #00ff00; }
    div[data-testid="stExpander"] { background-color: #1e2530; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

st.title("QUANTITATIVE STRATEGY DESK :: RELATIVE ROTATION & PERFORMANCE")
st.markdown("---")

# 1. Caricamento Dati
st.sidebar.markdown("### DATA INGESTION")
uploaded_file = st.sidebar.file_uploader("Upload Time Series (.csv, .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if df.shape[1] < 2:
                uploaded_file.seek(0) 
                df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        else:
            df = pd.read_excel(uploaded_file)
            
        if df.shape[1] < 2:
            st.error("SYS_ERR: Invalid format. Require Date + Asset columns.")
            st.stop()

        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col)
        
        if df.map(lambda x: isinstance(x, str)).any().any():
            df = df.replace({',': '.'}, regex=True)
            
        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all').sort_index()

    except Exception as e:
        st.error(f"SYS_ERR: Parsing failure. {e}")
        st.stop()

    available_assets = df.columns.tolist()

    # 2. Setup Globale
    st.sidebar.markdown("### STRATEGY PARAMETERS")
    default_bench = "^GSPC" if "^GSPC" in available_assets else available_assets[0]
    benchmark = st.sidebar.selectbox("Market Benchmark", options=available_assets, index=available_assets.index(default_bench))
    
    rrg_options = [a for a in available_assets if a != benchmark]
    selected_assets = st.sidebar.multiselect("Target Assets", options=rrg_options, default=rrg_options[:4])
    
    rrg_window = st.sidebar.number_input("RRG Moving Avg Window", min_value=5, max_value=50, value=14)
    
    if not selected_assets:
        st.warning("AWAITING ASSET SELECTION.")
        st.stop()

    # Calcoli Preliminari per Metriche di Testa
    df_selected = df[[benchmark] + selected_assets].copy()
    last_date = df_selected.index[-1]
    prev_date = df_selected.index[-2] if len(df_selected) > 1 else last_date
    
    # Dashboard Metriche (Stile Trading Desk)
    st.markdown("#### MARKET SNAPSHOT (Latest: {})".format(last_date.strftime('%Y-%m-%d')))
    cols = st.columns(len(selected_assets) + 1)
    
    # Benchmark Metric
    b_ret = ((df_selected[benchmark].iloc[-1] / df_selected[benchmark].iloc[-2]) - 1) * 100
    cols[0].metric(label=f"BENCHMARK: {benchmark}", value=f"{df_selected[benchmark].iloc[-1]:.2f}", delta=f"{b_ret:.2f}%")
    
    # Asset Metrics
    for i, asset in enumerate(selected_assets):
        a_ret = ((df_selected[asset].iloc[-1] / df_selected[asset].iloc[-2]) - 1) * 100
        cols[i+1].metric(label=asset, value=f"{df_selected[asset].iloc[-1]:.2f}", delta=f"{a_ret:.2f}%")

    st.markdown("---")

    tab1, tab2 = st.tabs(["RELATIVE ROTATION GRAPH (RRG) & HEATMAP", "ABSOLUTE PERFORMANCE (PRICE ACTION)"])

    with tab2:
        st.markdown("### HISTORICAL PRICE ACTION")
        col1, col2 = st.columns([3, 1])
        with col2:
            chart_style = st.selectbox("Render Mode", options=["Lines", "Lines+Markers", "Markers"], index=0)
            normalize = st.checkbox("Base 100 Normalization", value=True)
            
        start_date = st.date_input("Start Date", value=df_selected.index.min().date())
        end_date = st.date_input("End Date", value=df_selected.index.max().date())
        
        df_trad = df_selected.loc[start_date:end_date]
        if normalize:
            df_trad = (df_trad / df_trad.iloc[0]) * 100
            
        df_melted = df_trad.reset_index().melt(id_vars=date_col, var_name='Asset', value_name='Value')
        mode_map = {"Lines": "lines", "Lines+Markers": "lines+markers", "Markers": "markers"}
        
        fig_trad = px.line(df_melted, x=date_col, y='Value', color='Asset', template="plotly_dark")
        fig_trad.update_traces(mode=mode_map[chart_style])
        fig_trad.update_layout(hovermode="x unified", height=600, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_trad, use_container_width=True)

    with tab1:
        col_rrg, col_heat = st.columns([1.2, 1])
        
        # Calcolo RRG Engine
        df_rrg = df_selected.copy()
        for asset in selected_assets:
            df_rrg[f'RS_{asset}'] = df_rrg[asset] / df_rrg[benchmark]
        for asset in selected_assets:
            rs_sma = df_rrg[f'RS_{asset}'].rolling(window=rrg_window).mean()
            df_rrg[f'Ratio_{asset}'] = 100 * (df_rrg[f'RS_{asset}'] / rs_sma)
        for asset in selected_assets:
            ratio_sma = df_rrg[f'Ratio_{asset}'].rolling(window=rrg_window).mean()
            df_rrg[f'Mom_{asset}'] = 100 * (df_rrg[f'Ratio_{asset}'] / ratio_sma)

        df_rrg_clean = df_rrg.dropna()

        if df_rrg_clean.empty:
            st.error(f"SYS_ERR: Insufficient historical data for {rrg_window}-period SMA.")
        else:
            with col_rrg:
                st.markdown("### RELATIVE ROTATION GRAPH")
                col_slider, col_toggle = st.columns(2)
                with col_slider:
                    tail_length = st.slider("Tail Length (Periods)", min_value=1, max_value=20, value=5)
                with col_toggle:
                    show_tail = st.toggle("Render Trail (Uncheck for static slide export)", value=True)

                df_tail = df_rrg_clean.tail(tail_length)
                
                plot_data = []
                for asset in selected_assets:
                    for i, date in enumerate(df_tail.index):
                        plot_data.append({
                            'Date': date, 'Asset': asset,
                            'RS-Ratio': df_tail.loc[date, f'Ratio_{asset}'],
                            'RS-Momentum': df_tail.loc[date, f'Mom_{asset}'],
                            'Is_Current': "Current" if i == len(df_tail) - 1 else "Historical"
                        })
                
                df_plot = pd.DataFrame(plot_data)
                current_points = df_plot[df_plot['Is_Current'] == "Current"]
                
                if show_tail:
                    fig_rrg = px.line(df_plot, x='RS-Ratio', y='RS-Momentum', color='Asset', markers=True, hover_data=['Date'], template="plotly_dark")
                    fig_rrg.add_scatter(x=current_points['RS-Ratio'], y=current_points['RS-Momentum'], mode='markers', marker=dict(size=14, color='white', line=dict(width=2, color='black')), showlegend=False, hoverinfo='skip')
                else:
                    fig_rrg = px.scatter(current_points, x='RS-Ratio', y='RS-Momentum', color='Asset', hover_data=['Date'], template="plotly_dark")
                    fig_rrg.update_traces(marker=dict(size=14, line=dict(width=2, color='white')))

                fig_rrg.add_vline(x=100, line_dash="dash", line_color="#555555")
                fig_rrg.add_hline(y=100, line_dash="dash", line_color="#555555")
                
                fig_rrg.add_annotation(x=102, y=102, text="LEADING", showarrow=False, font=dict(color="#00FF00", size=10))
                fig_rrg.add_annotation(x=98, y=102, text="IMPROVING", showarrow=False, font=dict(color="#00BFFF", size=10))
                fig_rrg.add_annotation(x=102, y=98, text="WEAKENING", showarrow=False, font=dict(color="#FFD700", size=10))
                fig_rrg.add_annotation(x=98, y=98, text="LAGGING", showarrow=False, font=dict(color="#FF0000", size=10))

                fig_rrg.update_layout(
                    xaxis_title="RS-Ratio (Trend)", yaxis_title="RS-Momentum (Velocity)",
                    height=600, margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_rrg, use_container_width=True)

            with col_heat:
                st.markdown("### PERIODIC RETURN MATRIX")
                hm_periods = min(15, len(df_rrg_clean))
                df_returns = df_rrg_clean[selected_assets].pct_change().dropna() * 100
                df_hm = df_returns.tail(hm_periods).T
                df_hm.columns = df_hm.columns.strftime('%m-%d')

                fig_hm = px.imshow(
                    df_hm,
                    labels=dict(x="Period", y="Asset", color="Ret %"),
                    x=df_hm.columns, y=df_hm.index,
                    text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdYlGn", template="plotly_dark"
                )
                fig_hm.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_hm, use_container_width=True)

        # Motore Logico (Compresso)
        st.markdown("---")
        with st.expander("QUANTITATIVE STRATEGY SUMMARY (AUTO-GENERATED)", expanded=True):
            if not df_returns.empty:
                last_period_returns = df_returns.iloc[-1]
                for asset in selected_assets:
                    if asset in df_returns.columns and not current_points[current_points['Asset'] == asset].empty:
                        ret = last_period_returns[asset]
                        row = current_points[current_points['Asset'] == asset].iloc[0]
                        rs_ratio, rs_mom = row['RS-Ratio'], row['RS-Momentum']
                        
                        if rs_ratio >= 100 and rs_mom >= 100: quad = "LEADING"
                        elif rs_ratio < 100 and rs_mom >= 100: quad = "IMPROVING"
                        elif rs_ratio >= 100 and rs_mom < 100: quad = "WEAKENING"
                        else: quad = "LAGGING"
                        
                        st.markdown(f"**[{asset}]** :: RRG: `{quad}` | Last Return: `{ret:+.2f}%`")
