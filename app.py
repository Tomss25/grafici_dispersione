import streamlit as st
import pandas as pd
import plotly.express as px

# Configurazione della pagina
st.set_page_config(page_title="Analisi Serie Storiche", layout="wide")
st.title("Analisi e Confronto Serie Storiche Finanziarie")

# 1. Gestione dell'Input (Dati)
st.sidebar.header("1. Caricamento Dati")
uploaded_file = st.sidebar.file_uploader("Carica file (.csv, .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Lettura del file in base all'estensione e formato
        if uploaded_file.name.endswith('.csv'):
            # Tentativo 1: Formato Standard Internazionale (separatore virgola)
            df = pd.read_csv(uploaded_file)
            
            # Se ha letto una sola colonna, è quasi certamente il formato Europeo
            if df.shape[1] < 2:
                uploaded_file.seek(0) # Riavvolge il buffer del file
                # Tentativo 2: Formato Europeo (separatore punto e virgola, decimali con virgola)
                df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        else:
            df = pd.read_excel(uploaded_file)
            
        # Controllo struttura minima
        if df.shape[1] < 2:
            st.error("Il file deve contenere almeno due colonne: Data e almeno un Asset.")
            st.stop()

        # Parsing della prima colonna come DateTime
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Rimozione righe con date non valide
        if df[date_col].isnull().any():
            st.warning("Alcune date non sono state riconosciute e sono state scartate.")
            df = df.dropna(subset=[date_col])
            
        df = df.set_index(date_col)
        
        # Pulisci le eventuali virgole rimaste come stringhe e forza il tipo numerico
        # Questo previene crash se pandas non ha convertito correttamente i decimali
        if df.map(lambda x: isinstance(x, str)).any().any():
            df = df.replace({',': '.'}, regex=True)
            
        df = df.apply(pd.to_numeric, errors='coerce') 
        df = df.dropna(how='all') 
        df = df.sort_index() 

    except Exception as e:
        # L'indentazione qui è vitale. Questo blocco gestisce tutto il try superiore.
        st.error(f"Errore critico durante il parsing del file. Dettagli: {e}")
        st.stop()

    # 2. Interfaccia Utente e Filtri
    st.sidebar.header("2. Filtri di Analisi")
    
    # Selezione Asset
    available_assets = df.columns.tolist()
    selected_assets = st.sidebar.multiselect(
        "Seleziona Asset",
        options=available_assets,
        default=available_assets
    )
    
    if not selected_assets:
        st.warning("Seleziona almeno un asset per procedere.")
        st.stop()
        
    df_selected = df[selected_assets]

    # Orizzonte temporale
    min_date = df_selected.index.min().date()
    max_date = df_selected.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Orizzonte Temporale",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) != 2:
        st.stop()
        
    start_date, end_date = date_range
    mask = (df_selected.index.date >= start_date) & (df_selected.index.date <= end_date)
    df_filtered = df_selected.loc[mask]

    # Timeframe (Resampling)
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=["Giornaliero", "Settimanale", "Mensile"]
    )
    
    # Logica di resampling finanziario (ultimo prezzo del periodo)
    if timeframe == "Settimanale":
        df_filtered = df_filtered.resample('W').last()
    elif timeframe == "Mensile":
        df_filtered = df_filtered.resample('ME').last()

    # Normalizzazione Base 100: Trasforma i prezzi in percentuali relative al punto di partenza
    normalize = st.sidebar.checkbox("Normalizza a Base 100", value=True, help="Mostra la performance relativa permettendo di confrontare asset con prezzi assoluti molto diversi.")
    
    if normalize:
        # Divide tutto per la prima riga e moltiplica per 100
        df_filtered = (df_filtered / df_filtered.iloc[0]) * 100
        y_label = "Valore Normalizzato (Base 100)"
    else:
        y_label = "Prezzo Assoluto"

    # Trasformazione dei dati in formato long per Plotly
    df_melted = df_filtered.reset_index().melt(
        id_vars=date_col, 
        value_vars=selected_assets, 
        var_name='Asset', 
        value_name='Prezzo'
    )

    # 3. Output (Visualizzazione)
    st.subheader(f"Evoluzione Asset ({timeframe})")
    
    fig = px.line(
        df_melted, 
        x=date_col, 
        y='Prezzo', 
        color='Asset',
        markers=True, 
        title="Confronto Performance",
        hover_data={"Prezzo": ":.2f", date_col: "|%Y-%m-%d"}
    )
    
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title=y_label,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Visualizza i dati grezzi analizzati"):
        st.dataframe(df_filtered)

else:
    st.info("Attesa caricamento file. Usa la barra laterale.")
