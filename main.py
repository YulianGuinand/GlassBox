import sys
import os
import torch

# Add 'src' to sys.path to allow imports from 'glassbox' package
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import streamlit as st
import time
import queue
from glassbox.ui.layout import render_layout
from glassbox.utils.session import SessionManager
from glassbox.utils.threading import TrainingThread, init_training_queue
from glassbox.core.data_manager import DataManager
from glassbox.core.model_factory import ModelFactory
from glassbox.core.trainer import training_worker
from glassbox.config import DEVICE

def main():
    # 1. Init Session State
    SessionManager.init_state("metrics_history", [])
    SessionManager.init_state("is_training", False)
    SessionManager.init_state("training_thread", None)
    SessionManager.init_state("trained_model", None) # Store model for inference
    
    # 2. Render Layout
    # Layout now returns (config, inputs) tuple
    result = render_layout()
    config_params, inference_inputs = result if result else (None, None)
    
    # 3. Handle Start Training
    if config_params and not inference_inputs and not SessionManager.get("is_training"):
        dm = SessionManager.get_data_manager()
        
        if not dm or dm.X_train is None:
            st.error("❌ Veuillez d'abord charger et prétraiter les données (Onglet 1).")
        else:
            # Create Model
            model = ModelFactory.create_mlp(
                input_dim=dm.input_dim,
                hidden_layers=config_params['hidden_layers'],
                output_dim=dm.output_dim,
                activation=config_params['activation'],
                dropout_rate=config_params['dropout']
            )
            
            # Save model ref for training
            SessionManager.set("current_model", model)
            
            # Create Loaders
            train_loader, test_loader = dm.get_dataloaders(batch_size=int(config_params['batch_size']))
            
            # Prepare Training Params
            criterion = 'CrossEntropyLoss' if dm.is_classification else 'MSELoss'
            
            train_params = {
                'model': model,
                'train_loader': train_loader,
                'test_loader': test_loader,
                'epochs': int(config_params['epochs']),
                'learning_rate': config_params['lr'],
                'optimizer': 'Adam',
                'criterion': criterion,
                'is_classification': dm.is_classification
            }
            
            # Start Background Thread
            ts_queue = init_training_queue()
            SessionManager.set("training_queue", ts_queue)
            
            thread = TrainingThread(training_worker, ts_queue, train_params)
            thread.start()
            
            SessionManager.set("training_thread", thread)
            SessionManager.set("is_training", True)
            SessionManager.set("metrics_history", []) 
            st.rerun()

    # 4. Handle Inference
    if inference_inputs:
        model = SessionManager.get("trained_model")
        dm = SessionManager.get_data_manager()
        if model and dm:
            try:
                # Transform inputs
                X_tensor = dm.transform_input(inference_inputs).to(DEVICE)
                
                # Predict
                model.eval()
                with torch.no_grad():
                    output = model(X_tensor)
                    
                    if dm.is_classification:
                        # Softmax for prob
                        probs = torch.nn.functional.softmax(output, dim=1)
                        pred_idx = torch.argmax(probs, dim=1).item()
                        
                        # Decode label
                        label = dm.label_encoder.inverse_transform([pred_idx])[0]
                        confidence = probs[0][pred_idx].item()
                        
                        st.success(f"### Prédiction : **{label}**")
                        st.info(f"Confiance : {confidence:.2%}")
                    else:
                        pred_val = output.item()
                        st.success(f"### Prédiction : **{pred_val:.4f}**")
                        
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

    # 5. Handle Training Loop (Polling)
    if SessionManager.get("is_training"):
        ts_queue = SessionManager.get_training_queue()
        try:
            # Read all available messages
            while True:
                msg = ts_queue.get_nowait()
                
                if msg['status'] == 'running':
                    # Append metrics
                    history = SessionManager.get("metrics_history")
                    history.append(msg)
                    SessionManager.set("metrics_history", history)
                    
                elif msg['status'] == 'finished':
                    SessionManager.set("is_training", False)
                    SessionManager.set("training_thread", None)
                    
                    # Store tuned model
                    trained_model = SessionManager.get("current_model")
                    SessionManager.set("trained_model", trained_model)
                    
                    st.success("✅ Entraînement Terminé ! Modèle sauvegardé.")
                    st.rerun() 
                    break
                    
                elif msg['status'] == 'error':
                    SessionManager.set("is_training", False)
                    st.error(f"❌ Erreur d'entraînement : {msg['message']}")
                    break
                    
        except queue.Empty:
            pass
            
        # Rerun to update the UI
        time.sleep(0.1) 
        st.rerun()

if __name__ == "__main__":
    main()
