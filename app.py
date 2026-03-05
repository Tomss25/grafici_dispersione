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
        # Lettura del file in base all'estensione
        if uploaded_file.name.endswith('.csv'):
            # Tentativo 1: Standard Internazionale (separatore ,)
            df = pd.read_csv(uploaded_file)
            
            # Se ha letto una sola colonna, probabile formato Europeo
            if df.shape[1] < 2:
                uploaded_file.seek(0) # Riavvolge il file per rileggerlo
                # Tentativo 2: Formato Europeo (separatore ; e decimale ,)
                df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        else:
            df = pd.read_excel(uploaded_file)
            
        # Controllo struttura minima dopo il parsing corretto
        if df.shape[1] < 2:
            st.error("Il file deve contenere almeno due colonne: Data e almeno un Asset. Verifica il delimitatore del tuo CSV.")
            st.stop()

        # Parsing della prima colonna come DateTime
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Rimozione righe con date non valide
        if df[date_col].isnull().any():
            st.warning("Alcune date non sono state riconosciute e sono state scartate.")
            df = df.dropna(subset=[date_col])
            
        df = df.set_index(date_col)
        
        # Sostituiamo eventuali virgole rimaste come stringhe (se il file era misto) prima di forzare a numerico
        df = df.replace({',': '.'}, regex=True)
        df = df.apply(pd.to_numeric, errors='coerce') # Forza i prezzi a numerico
        
        df = df.dropna(how='all') # Rimuovi righe completamente vuote
        df = df.sort_index() # Assicura l'ordine cronologico

    except Exception as e:
        st.error(f"Errore critico durante il parsing del file. Verifica la formattazione. Dettagli: {e}")
        st.stop()

    # 2. Interfaccia Utente e Filtri
    st.sidebar.header("2. Filtri di Analisi")
    
    # Selezione Asset
    available_assets = df.columns.tolist()
    selected_assets = st.sidebar.multiselect(
        "Seleziona Asset da analizzare",
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
        "Seleziona Orizzonte Temporale",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) != 2:
        st.stop() # Attende che l'utente selezioni entrambe le date
        
    start_date, end_date = date_range
    mask = (df_selected.index.date >= start_date) & (df_selected.index.date <= end_date)
    df_filtered = df_selected.loc[mask]

    # Timeframe (Resampling)
    timeframe = st.sidebar.selectbox(
        "Seleziona Timeframe",
        options=["Giornaliero (Originale)", "Settimanale", "Mensile"]
    )
    
    # Logica di resampling: usiamo l'ultimo valore del periodo (chiusura), non la media
    if timeframe == "Settimanale":
        df_filtered = df_filtered.resample('W').last()
    elif timeframe == "Mensile":
        df_filtered = df_filtered.resample('ME').last()

    # Trasformazione dei dati in formato long per Plotly
    df_melted = df_filtered.reset_index().melt(
        id_vars=date_col, 
        value_vars=selected_assets, 
        var_name='Asset', 
        value_name='Prezzo'
    )

    # 3. Output (Visualizzazione)
    st.subheader(f"Evoluzione Prezzi ({timeframe})")
    
    # Generazione grafico: Uso un grafico a linee con marcatori (scatter)
    # per unire la tua richiesta di "scatter" con la necessità logica di una linea temporale
    fig = px.line(
        df_melted, 
        x=date_col, 
        y='Prezzo', 
        color='Asset',
        markers=True, # Inserisce i punti (scatter) sulle linee
        title="Confronto Asset Selezionati",
        hover_data={"Prezzo": ":.2f", date_col: "|%Y-%m-%d"}
    )
    
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Prezzo Assoluto",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostra i dati tabulari su richiesta
    with st.expander("Visualizza i dati grezzi filtrati"):
        st.dataframe(df_filtered)

else:

    st.info("Attesa caricamento file. Usa la barra laterale per iniziare.")
