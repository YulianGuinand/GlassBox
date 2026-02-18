import streamlit as st
from glassbox.utils.session import SessionManager
from glassbox.core.model_factory import ModelFactory
from glassbox.ui.styles import load_custom_css, card_begin, card_end
from glassbox.ui.components import (
    render_data_upload, 
    render_model_config, 
    render_training_config,
    render_prediction_inputs,
    render_export_section
)
from glassbox.ui.visuals import (
    render_network_schema, 
    render_metrics,
    render_evaluation_report
)

def render_layout():
    """
    Renders the modern GlassBox layout.
    """
    # 1. Load CSS
    load_custom_css()
    
    # 2. Header
    st.markdown("<h1><span class='highlight'>GlassBox</span> Studio</h1>", unsafe_allow_html=True)
    st.caption("Plateforme interactive de Réseaux de Neurones")
    
    # TABS STRUCTURE (No emojis)
    tab_data, tab_arch, tab_train, tab_pred = st.tabs(["Données", "Architecture", "Entraînement", "Prédiction"])
    
    config_params = {}
    inference_inputs = None
    dm = SessionManager.get_data_manager()
    
    # --- TAB 1: DONNÉES ---
    with tab_data:
        card_begin()
        render_data_upload()
        card_end()
    
    # --- TAB 2: ARCHITECTURE ---
    with tab_arch:
        col_conf, col_viz = st.columns([1, 2])
        
        with col_conf:
            card_begin()
            hidden_layers, activation, dropout = render_model_config()
            card_end()
            
            config_params.update({
                "hidden_layers": hidden_layers,
                "activation": activation,
                "dropout": dropout
            })
            
        with col_viz:
            st.markdown("### Visualisation")
            if dm and dm.input_dim > 0:
                preview_model = ModelFactory.create_mlp(
                    input_dim=dm.input_dim,
                    hidden_layers=hidden_layers,
                    output_dim=dm.output_dim,
                    activation=activation,
                    dropout_rate=dropout
                )
                render_network_schema(preview_model)
            else:
                st.warning("Veuillez charger des données pour visualiser le réseau.")

    # --- TAB 3: ENTRAÎNEMENT ---
    with tab_train:
        col_param, col_start = st.columns([2, 1])
        
        with col_param:
            card_begin()
            lr, epochs, batch_size = render_training_config()
            card_end()
            
            config_params.update({
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size
            })
            
        with col_start:
            st.markdown("### Contrôle")
            start_btn = st.button("Lancer l'Entraînement", type="primary", disabled=SessionManager.get("is_training"))
            
            if SessionManager.get("is_training"):
                st.spinner("Calcul en cours...")
        
        st.divider()
        
        # Metrics Area
        metrics = SessionManager.get("metrics_history", [])
        is_classif = dm.is_classification if dm else True
        
        if metrics:
            card_begin()
            render_metrics(metrics, is_classification=is_classif)
            card_end()
        else:
            st.info("En attente de démarrage...")

    # --- TAB 4: PRÉDICTION ---
    with tab_pred:
        model_state = SessionManager.get("trained_model")
        
        if not model_state:
             st.warning("Veuillez d'abord entraîner un modèle.")
        else:
             # Export
             card_begin()
             render_export_section(model_state)
             card_end()
             
             # Prediction
             if dm:
                 card_begin()
                 inputs = render_prediction_inputs(dm)
                 if st.button("Calculer la Prédiction", type="primary"):
                     return config_params, inputs
                 card_end()
                 
    if start_btn:
        return config_params, None
    
    return None, None
