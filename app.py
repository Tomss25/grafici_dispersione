import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Quantitative Strategy Desk", layout="wide")

# Iniezione CSS (Mantenuto il tuo stile moderno/Apple)
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div,
        .stTextInput input, .stNumberInput input, .stDateInput input {
            border-radius: 12px !important;
        }
        div[data-testid="stAlert"] {
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        }
        button[data-baseweb="tab"] {
            border-radius: 10px 10px 0px 0px !important;
        }
        div[data-testid="stExpander"] {
            border-radius: 12px !important;
            border: 1px solid #eaeaea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title("Analisi Serie Storiche & RRG (Z-Score Model)")

st.sidebar.header("1. Dati")
uploaded_file = st.sidebar.file_uploader("Carica file", type=["csv", "xlsx"])

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
            st.error("Formato non valido. Il file deve contenere Data e Asset.")
            st.stop()

        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col)
        
        if df.map(lambda x: isinstance(x, str)).any().any():
            df = df.replace({',': '.'}, regex=True)
            
        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all').sort_index()

    except Exception as e:
        st.error(f"Errore parsing: {e}")
        st.stop()

    available_assets = df.columns.tolist()

    tab1, tab2 = st.tabs(["Relative Rotation Graph (RRG) & Heatmap", "Analisi Tradizionale"])

    # ---------------------------------------------------------
    # TAB 1: RRG Z-SCORE ENGINE
    # ---------------------------------------------------------
    with tab1:
        st.header("Relative Rotation Graph (RRG) & Heatmap")
        
        col_bench, col_ass = st.columns([1, 2])
        with col_bench:
            default_bench = "^GSPC" if "^GSPC" in available_assets else available_assets[0]
            benchmark = st.selectbox("Benchmark", options=available_assets, index=available_assets.index(default_bench))
        with col_ass:
            rrg_options = [a for a in available_assets if a != benchmark]
            selected_rrg_assets = st.multiselect("Asset da analizzare", options=rrg_options, default=rrg_options[:3])
            
        st.sidebar.markdown("---")
        st.sidebar.header("Impostazioni Visualizzazione")
        tail_length = st.sidebar.slider("Lunghezza Coda (Ultimi N periodi)", min_value=1, max_value=20, value=5)

        # Alert brutale sui dati necessari
        st.info("📊 **Motore Z-Score Attivo:** Il calcolo richiede EMA(12) + EMA(26) + Rolling(52) + Delta(1) + Rolling(14). Le prime ~70 righe del dataset verranno consumate per il warm-up statistico.")

        if selected_rrg_assets:
            df_rrg = df[[benchmark] + selected_rrg_assets].copy()
            
            for asset in selected_rrg_assets:
                # 1. Input e RS_raw = Settore / SP500
                rs_raw = df_rrg[asset] / df_rrg[benchmark]
                
                # 2. EMA12(RS_raw)
                ema_12 = rs_raw.ewm(span=12, adjust=False).mean()
                
                # 3. EMA26(EMA12) = RS_s (Doppio Smoothing)
                rs_s = ema_12.ewm(span=26, adjust=False).mean()
                
                # 4. Z-score rolling 52w -> RS-Ratio (X) = 100 ± 10σ
                rolling_mean_52 = rs_s.rolling(window=52).mean()
                rolling_std_52 = rs_s.rolling(window=52).std()
                z_score_ratio = (rs_s - rolling_mean_52) / rolling_std_52
                df_rrg[f'Ratio_{asset}'] = 100 + (z_score_ratio * 10)
                
                # 5. ΔRS-Ratio -> Z-score rolling 14w -> RS-Momentum (Y) = 100 ± 10σ
                delta_rs_ratio = df_rrg[f'Ratio_{asset}'].diff(1)
                rolling_mean_14_mom = delta_rs_ratio.rolling(window=14).mean()
                rolling_std_14_mom = delta_rs_ratio.rolling(window=14).std()
                z_score_mom = (delta_rs_ratio - rolling_mean_14_mom) / rolling_std_14_mom
                df_rrg[f'Mom_{asset}'] = 100 + (z_score_mom * 10)

            # Rimuoviamo tutti i NaN generati dal tremendo consumo di dati delle finestre rolling
            df_rrg_clean = df_rrg.dropna()

            if df_rrg_clean.empty or len(df_rrg_clean) < tail_length:
                st.error("🚨 **ERRORE CRITICO DEI DATI:** Non hai abbastanza storico. Il modello Rolling Z-Score ha bruciato i dati per calcolare le deviazioni standard e non è rimasto nulla da plottare. Carica un file con almeno 2-3 anni di storico continuo.")
            else:
                df_tail = df_rrg_clean.tail(tail_length)
                
                plot_data = []
                for asset in selected_rrg_assets:
                    for i, date in enumerate(df_tail.index):
                        plot_data.append({
                            'Data': date,
                            'Asset': asset,
                            'RS-Ratio': df_tail.loc[date, f'Ratio_{asset}'],
                            'RS-Momentum': df_tail.loc[date, f'Mom_{asset}'],
                            'Is_Current': "Ultimo" if i == len(df_tail) - 1 else "Storico"
                        })
                
                df_plot = pd.DataFrame(plot_data)
                current_points = df_plot[df_plot['Is_Current'] == "Ultimo"]
                
                st.markdown("---")
                show_tail = st.toggle("Mostra scia di rotazione (Storico)", value=True)
                
                if show_tail:
                    fig_rrg = px.line(df_plot, x='RS-Ratio', y='RS-Momentum', color='Asset', markers=True, hover_data=['Data'])
                    fig_rrg.add_scatter(x=current_points['RS-Ratio'], y=current_points['RS-Momentum'], mode='markers', marker=dict(size=12, color='black'), showlegend=False, hoverinfo='skip')
                else:
                    fig_rrg = px.scatter(current_points, x='RS-Ratio', y='RS-Momentum', color='Asset', hover_data=['Data'])
                    fig_rrg.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))

                fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray")
                fig_rrg.add_hline(y=100, line_dash="dash", line_color="gray")
                
                fig_rrg.add_annotation(x=102, y=102, text="Leading", showarrow=False, font=dict(color="green"))
                fig_rrg.add_annotation(x=98, y=102, text="Improving", showarrow=False, font=dict(color="blue"))
                fig_rrg.add_annotation(x=102, y=98, text="Weakening", showarrow=False, font=dict(color="orange"))
                fig_rrg.add_annotation(x=98, y=98, text="Lagging", showarrow=False, font=dict(color="red"))

                # Fissiamo i limiti degli assi per dare stabilità visiva al grafico Z-Score
                fig_rrg.update_layout(
                    title=f"Relative Rotation Graph (Z-Score Model) vs {benchmark}",
                    xaxis_title="RS-Ratio (Forza Relativa Z-Score)",
                    yaxis_title="RS-Momentum (Spinta Z-Score)",
                    height=700,
                    width=700,
                    xaxis=dict(range=[70, 130]), # +/- 3 Deviazioni Standard (100 ± 30)
                    yaxis=dict(range=[70, 130])
                )
                
                col_left, col_center, col_right = st.columns([1, 3, 1])
                with col_center:
                    st.plotly_chart(fig_rrg, use_container_width=True)

                st.markdown("### Analisi Dinamica Heatmap")
                hm_periods = min(12, len(df_rrg_clean))
                df_returns = df_rrg_clean[selected_rrg_assets].pct_change().dropna() * 100
                df_hm = df_returns.tail(hm_periods).T
                df_hm.columns = df_hm.columns.strftime('%Y-%m-%d')

                fig_hm = px.imshow(
                    df_hm,
                    labels=dict(x="Data", y="Asset", color="Rendimento %"),
                    x=df_hm.columns,
                    y=df_hm.index,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                )
                fig_hm.update_layout(height=400)
                st.plotly_chart(fig_hm, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: ANALISI TRADIZIONALE
    # ---------------------------------------------------------
    with tab2:
        st.header("Grafico Prezzi Storici")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_assets_trad = st.multiselect("Asset", options=available_assets, default=available_assets[:3], key="trad_assets")
        with col2:
            chart_style = st.selectbox("Stile Grafico", options=["Solo Linee", "Linee + Punti", "Solo Punti"], index=0, key="chart_style")

        if not selected_assets_trad:
            st.warning("Seleziona asset.")
        else:
            df_trad = df[selected_assets_trad]
            
            start_date_trad = st.date_input("Inizio", value=df_trad.index.min().date(), key="start_trad")
            end_date_trad = st.date_input("Fine", value=df_trad.index.max().date(), key="end_trad")
            
            df_trad = df_trad.loc[start_date_trad:end_date_trad]
            
            if st.checkbox("Normalizza a Base 100", value=True):
                df_trad = (df_trad / df_trad.iloc[0]) * 100
                
            df_melted = df_trad.reset_index().melt(id_vars=date_col, var_name='Asset', value_name='Valore')

            mode_map = {"Solo Linee": "lines", "Linee + Punti": "lines+markers", "Solo Punti": "markers"}
            
            fig = px.line(df_melted, x=date_col, y='Valore', color='Asset')
            fig.update_traces(mode=mode_map[chart_style])
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
