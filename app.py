import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Quantitative Strategy Desk", layout="wide")

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
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
                
            if df.shape[1] < 2:
                uploaded_file.seek(0) 
                try:
                    df = pd.read_csv(uploaded_file, sep=';', decimal=',')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file)
            
        if df.shape[1] < 2:
            st.error("Formato non valido. Il file deve contenere Data e Asset.")
            st.stop()

        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col)
        
        if hasattr(df, 'map'):
            if df.map(lambda x: isinstance(x, str)).any().any():
                df = df.replace({',': '.'}, regex=True)
        else:
            if df.applymap(lambda x: isinstance(x, str)).any().any():
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
        st.sidebar.header("Architettura Modello")
        
        data_freq = st.sidebar.selectbox(
            "Frequenza Dati Input", 
            options=["Settimanale", "Giornaliero", "Mensile"],
            help="Modifica i pesi del Z-Score per mantenere la finestra di osservazione pari a 1 Anno (Ratio) e 1 Trimestre (Momentum)."
        )
        
        if data_freq == "Giornaliero":
            span1, span2 = 60, 130
            win_ratio, min_ratio = 252, 63
            win_mom, min_mom = 70, 21
            st.sidebar.info("⚙️ Parametri: EMA(60/130), Z-Score 1A(252g), Mom 1T(70g)")
        elif data_freq == "Settimanale":
            span1, span2 = 12, 26
            win_ratio, min_ratio = 52, 14
            win_mom, min_mom = 14, 7
            st.sidebar.info("⚙️ Parametri: EMA(12/26), Z-Score 1A(52s), Mom 1T(14s)")
        else: # Mensile
            span1, span2 = 3, 6
            win_ratio, min_ratio = 12, 6
            win_mom, min_mom = 3, 2
            st.sidebar.info("⚙️ Parametri: EMA(3/6), Z-Score 1A(12m), Mom 1T(3m)")

        st.sidebar.markdown("---")
        tail_length = st.sidebar.slider("Lunghezza Coda (Ultimi N periodi)", min_value=1, max_value=20, value=5)

        if selected_rrg_assets:
            df_rrg = df[[benchmark] + selected_rrg_assets].copy()
            
            for asset in selected_rrg_assets:
                # 1. RS_raw
                rs_raw = df_rrg[asset] / df_rrg[benchmark]
                
                # 2. EMA(span1)
                ema_1 = rs_raw.ewm(span=span1, adjust=False).mean()
                
                # 3. EMA(span2) su EMA(span1) = Doppio Smoothing
                rs_s = ema_1.ewm(span=span2, adjust=False).mean()
                
                # 4. Z-score rolling -> RS-Ratio (X)
                rolling_mean_ratio = rs_s.rolling(window=win_ratio, min_periods=min_ratio).mean()
                rolling_std_ratio = rs_s.rolling(window=win_ratio, min_periods=min_ratio).std()
                z_score_ratio = (rs_s - rolling_mean_ratio) / rolling_std_ratio
                df_rrg[f'Ratio_{asset}'] = 100 + (z_score_ratio * 10)
                
                # 5. ΔRS-Ratio -> Z-score rolling -> RS-Momentum (Y)
                delta_rs_ratio = df_rrg[f'Ratio_{asset}'].diff(1)
                rolling_mean_mom = delta_rs_ratio.rolling(window=win_mom, min_periods=min_mom).mean()
                rolling_std_mom = delta_rs_ratio.rolling(window=win_mom, min_periods=min_mom).std()
                z_score_mom = (delta_rs_ratio - rolling_mean_mom) / rolling_std_mom
                df_rrg[f'Mom_{asset}'] = 100 + (z_score_mom * 10)

            df_rrg_clean = df_rrg.dropna()

            if df_rrg_clean.empty or len(df_rrg_clean) < tail_length:
                st.error(f"🚨 **ERRORE DATI:** Non hai sufficiente storico per inizializzare il modello sul time frame '{data_freq}'. Aggiungi più righe al tuo file Excel.")
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

                fig_rrg.update_layout(
                    title=f"Relative Rotation Graph (Z-Score Model) vs {benchmark}",
                    xaxis_title="RS-Ratio (Forza Relativa Z-Score)",
                    yaxis_title="RS-Momentum (Spinta Z-Score)",
                    height=700,
                    width=700,
                    xaxis=dict(range=[70, 130]), 
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
                # SEZIONE REINSERITA: Analisi e Comparazione Automatica
                # ---------------------------------------------------------
                st.markdown("---")
                st.markdown("### Sintesi Strategica: RRG (Z-Score) Incrociato con Heatmap")
                
                if not df_returns.empty:
                    last_period_returns = df_returns.iloc[-1]
                    for asset in selected_rrg_assets:
                        if asset in df_returns.columns and not current_points[current_points['Asset'] == asset].empty:
                            ret = last_period_returns[asset]
                            row = current_points[current_points['Asset'] == asset].iloc[0]
                            rs_ratio = row['RS-Ratio']
                            rs_mom = row['RS-Momentum']
                            
                            # Logica dei quadranti pura
                            if rs_ratio >= 100 and rs_mom >= 100: quad = "Leading"
                            elif rs_ratio < 100 and rs_mom >= 100: quad = "Improving"
                            elif rs_ratio >= 100 and rs_mom < 100: quad = "Weakening"
                            else: quad = "Lagging"
                            
                            # Sintesi Brutale
                            if quad == "Leading" and ret > 0:
                                st.success(f"**{asset}**: VERO LEADER. È in *Leading* e ha generato capitale assoluto (+{ret:.2f}%). Sta trainando in termini sia relativi che assoluti.")
                            elif quad == "Leading" and ret <= 0:
                                st.error(f"**{asset}**: FALSO LEADER. È in *Leading* ma sta perdendo soldi ({ret:.2f}%). Crolla più lentamente del benchmark, ma distrugge comunque il tuo capitale.")
                            elif quad == "Lagging" and ret > 0:
                                st.warning(f"**{asset}**: ZAVORRA FORTUNATA. È in *Lagging* ma guadagna (+{ret:.2f}%). Non ha forza propria, sta solo galleggiando perché l'intero mercato sale. Alla prima correzione, collasserà.")
                            elif quad == "Lagging" and ret <= 0:
                                st.error(f"**{asset}**: DISTRUTTORE DI VALORE. È in *Lagging* e sanguina profitti ({ret:.2f}%). Sottoperforma un mercato che già di suo scende. Tossico.")
                            elif quad == "Improving" and ret > 0:
                                st.info(f"**{asset}**: ACCUMULAZIONE SANA. È in *Improving* e macina utili (+{ret:.2f}%). Recupera forza relativa e genera cassa. Da monitorare per un ingresso.")
                            elif quad == "Improving" and ret <= 0:
                                st.warning(f"**{asset}**: RIMBALZO DEL GATTO MORTO? È in *Improving* ma chiude in rosso ({ret:.2f}%). Sembra recuperare solo perché scende meno violentemente del benchmark.")
                            elif quad == "Weakening" and ret > 0:
                                st.warning(f"**{asset}**: ESAURIMENTO RIALZISTA. È in *Weakening* ma ti dà ancora soldi (+{ret:.2f}%). Attenzione: la spinta direzionale sta morendo. Prepara la via d'uscita.")
                            elif quad == "Weakening" and ret <= 0:
                                st.error(f"**{asset}**: INVERSIONE CONFERMATA. È in *Weakening* e ha già iniziato a bruciare cassa ({ret:.2f}%). Il trend si è rotto. Taglia.")

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
