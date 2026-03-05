import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analisi Quantitativa", layout="wide")
st.title("Analisi Serie Storiche & RRG")

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

    tab1, tab2 = st.tabs(["Analisi Tradizionale", "Relative Rotation Graph (RRG)"])

    with tab1:
        st.header("Grafico Prezzi Storici")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_assets_trad = st.multiselect("Asset", options=available_assets, default=available_assets[:3], key="trad_assets")
        with col2:
            # Modifica: "Solo Linee" è ora l'opzione di default (index=0)
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

    with tab2:
        st.header("Relative Rotation Graph (RRG)")
        st.markdown("Mostra la rotazione di forza e momentum rispetto a un benchmark. **Nota:** Il calcolo consuma i primi periodi storici per le medie mobili.")
        
        col_bench, col_ass = st.columns([1, 2])
        with col_bench:
            default_bench = "^GSPC" if "^GSPC" in available_assets else available_assets[0]
            benchmark = st.selectbox("Benchmark", options=available_assets, index=available_assets.index(default_bench))
        with col_ass:
            rrg_options = [a for a in available_assets if a != benchmark]
            selected_rrg_assets = st.multiselect("Asset da analizzare", options=rrg_options, default=rrg_options[:3])
            
        rrg_window = st.number_input("Periodi Media Mobile (Standard: 14)", min_value=5, max_value=50, value=14)
        tail_length = st.slider("Lunghezza Coda (Ultimi N periodi da mostrare)", min_value=1, max_value=20, value=5)

        if selected_rrg_assets:
            df_rrg = df[[benchmark] + selected_rrg_assets].copy()
            
            for asset in selected_rrg_assets:
                df_rrg[f'RS_{asset}'] = df_rrg[asset] / df_rrg[benchmark]
                
            for asset in selected_rrg_assets:
                rs_sma = df_rrg[f'RS_{asset}'].rolling(window=rrg_window).mean()
                df_rrg[f'Ratio_{asset}'] = 100 * (df_rrg[f'RS_{asset}'] / rs_sma)
                
            for asset in selected_rrg_assets:
                ratio_sma = df_rrg[f'Ratio_{asset}'].rolling(window=rrg_window).mean()
                df_rrg[f'Mom_{asset}'] = 100 * (df_rrg[f'Ratio_{asset}'] / ratio_sma)

            df_rrg_clean = df_rrg.dropna()

            if df_rrg_clean.empty or len(df_rrg_clean) < tail_length:
                st.error(f"Dati insufficienti. L'RRG richiede almeno {rrg_window * 2} periodi storici consecutivi per calcolare le medie. Allarga il timeframe dei tuoi dati grezzi.")
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
                
                fig_rrg = px.line(df_plot, x='RS-Ratio', y='RS-Momentum', color='Asset', markers=True, hover_data=['Data'])
                
                current_points = df_plot[df_plot['Is_Current'] == "Ultimo"]
                fig_rrg.add_scatter(x=current_points['RS-Ratio'], y=current_points['RS-Momentum'], mode='markers', marker=dict(size=12, color='black'), showlegend=False, hoverinfo='skip')

                fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray")
                fig_rrg.add_hline(y=100, line_dash="dash", line_color="gray")
                
                fig_rrg.add_annotation(x=102, y=102, text="Leading", showarrow=False, font=dict(color="green"))
                fig_rrg.add_annotation(x=98, y=102, text="Improving", showarrow=False, font=dict(color="blue"))
                fig_rrg.add_annotation(x=102, y=98, text="Weakening", showarrow=False, font=dict(color="orange"))
                fig_rrg.add_annotation(x=98, y=98, text="Lagging", showarrow=False, font=dict(color="red"))

                fig_rrg.update_layout(
                    title=f"Relative Rotation Graph vs {benchmark}",
                    xaxis_title="RS-Ratio (Forza Relativa)",
                    yaxis_title="RS-Momentum (Spinta)",
                    height=700,
                    width=700
                )
                
                # Modifica: Utilizzo di colonne per accentrare il grafico RRG
                col_left, col_center, col_right = st.columns([1, 3, 1])
                with col_center:
                    st.plotly_chart(fig_rrg, use_container_width=True)
