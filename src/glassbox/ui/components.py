import streamlit as st
import pandas as pd
import torch
import io
from glassbox.utils.session import SessionManager
from glassbox.core.data_manager import DataManager
from glassbox.core.model_factory import ModelFactory
from glassbox.config import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS

def render_data_upload():
    """Renders file uploader and data preview (Clean style)."""
    
    st.markdown("### Importation des Données")
    st.caption("Format accepté : CSV. Détection automatique du type de tâche.")
    
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file:
        dm = SessionManager.get_data_manager()
        if not dm:
            dm = DataManager()
            SessionManager.set_data_manager(dm)
            
        try:
           uploaded_file.seek(0)
           df = dm.load_csv(uploaded_file)
           st.success(f"{len(df)} lignes chargées.")
           st.dataframe(df.head(), height=150)
           
           all_cols = df.columns.tolist()
           
           st.markdown("#### Configuration")
           
           col_tgt, col_feat = st.columns([1, 2])
           
           with col_tgt:
               target = st.selectbox(
                   "Colonne Cible (Target)", 
                   all_cols, 
                   index=len(all_cols)-1
               )
           
           with col_feat:
               potential_features = [c for c in all_cols if c != target]
               features = st.multiselect(
                   "Colonnes Caractéristiques (Features)", 
                   potential_features, 
                   default=potential_features
               )

           st.markdown("#### Type de Tâche")
           
           task_type = st.radio(
               "Mode", 
               ["Auto", "Classification", "Régression"],
               horizontal=True
           )
           
           if st.button("Prétraiter les Données"):
               if not features:
                   st.error("Sélectionnez au moins une caractéristique.")
               else:
                   with st.spinner("Traitement..."):
                       try:
                           dm.preprocess(target_column=target, feature_columns=features)
                           
                           if task_type == "Classification":
                               dm.is_classification = True
                           elif task_type == "Régression":
                               dm.is_classification = False
                           
                           st.success("Prêt.")
                           
                           c1, c2, c3 = st.columns(3)
                           c1.metric("Entrées", dm.input_dim)
                           c2.metric("Sorties", dm.output_dim)
                           c3.metric("Mode", "Classification" if dm.is_classification else "Régression")
                           
                       except Exception as e:
                           st.error(f"Erreur : {e}")

        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

def render_model_config():
    """Renders inputs for model architecture."""
    st.markdown("### Architecture")
    
    num_layers = st.number_input("Nombre de Couches Cachées", min_value=1, max_value=10, value=2)
    hidden_layers = []
    
    for i in range(num_layers):
        col1, col2 = st.columns(2)
        with col1:
            size = st.slider(f"Taille Couche {i+1}", 4, 256, 64, key=f"l_{i}")
        with col2:
            dropout = st.slider(f"Dropout {i+1}", 0.0, 0.5, 0.0, step=0.1, key=f"d_{i}")
            
        hidden_layers.append(size)
    
    c_glob1, c_glob2 = st.columns(2)
    with c_glob1:
        dropout_global = st.slider("Dropout Global", 0.0, 0.5, 0.0)
    with c_glob2:
        activation = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"])
    
    return hidden_layers, activation, dropout_global

def render_training_config(is_classification: bool = True):
    """Renders training hyperparameters."""
    st.markdown("### Hyperparamètres")
    
    c1, c2 = st.columns(2)
    with c1:
        lr = st.number_input(
            "Learning Rate", 
            value=DEFAULT_LEARNING_RATE, 
            format="%.4f", 
            step=0.0001
        )
    with c2:
        epochs = st.number_input(
            "Époques", 
            value=DEFAULT_EPOCHS, 
            min_value=1
        )
        
    batch_size = st.number_input(
        "Taille du Batch", 
        value=DEFAULT_BATCH_SIZE
    )
    
    return lr, epochs, batch_size

def render_prediction_inputs(dm: DataManager):
    """
    Renders inputs for manual prediction.
    """
    st.markdown("### Simulation")
    st.caption("Configurez les valeurs d'entrée pour tester le modèle.")
    
    inputs = {}
    cols = st.columns(2)
    
    for i, col_name in enumerate(dm.feature_columns):
        dtype = getattr(dm, 'feature_types', {}).get(col_name, 'numeric')
        meta = getattr(dm, 'feature_metadata', {}).get(col_name, {})
        
        c = cols[i % 2]
        
        with c:
            if dtype == 'categorical':
                if col_name in dm.feature_encoders:
                    classes = dm.feature_encoders[col_name].classes_
                    default_idx = 0
                    if 'default' in meta and meta['default'] in classes:
                        import numpy as np
                        if meta['default'] in classes:
                             default_idx = int(np.where(classes == meta['default'])[0][0])
                        
                    val = st.selectbox(f"{col_name}", classes, index=default_idx)
                    inputs[col_name] = val
                else:
                    val = st.text_input(f"{col_name}")
                    inputs[col_name] = val
            
            elif dtype == 'date':
                default_date = meta.get('default', pd.Timestamp.today())
                min_date = meta.get('min', None)
                max_date = meta.get('max', None)
                
                val = st.date_input(
                    f"{col_name}", 
                    value=default_date,
                    min_value=min_date,
                    max_value=max_date
                )
                inputs[col_name] = val
                
            else:
                default_val = float(meta.get('default', 0.0))
                val = st.number_input(f"{col_name}", value=default_val, format="%.4f")
                inputs[col_name] = val
                
    return inputs

def render_export_section(model):
    """Renders download button."""
    if model:
        st.markdown("### Export")
        
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        
        st.download_button(
            label="Télécharger (.pth)",
            data=buffer,
            file_name="model.pth",
            mime="application/octet-stream"
        )
