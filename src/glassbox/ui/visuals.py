import streamlit as st
import graphviz
import pandas as pd
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from glassbox.config import COLOR_DENSE, COLOR_DROPOUT, COLOR_ACTIVATION, COLOR_NORM

def render_network_schema(model: nn.Module):
    """
    Visualizes the PyTorch model using Plotly for full interactivity (Zoom/Pan).
    """
    if model is None:
        st.info("Aucun modèle.")
        return

    # 1. Extract Architecture Topology
    layers_struct = []
    first_linear = True
    
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            if first_linear:
                layers_struct.append({"name": "Input", "size": layer.in_features, "color": "#FFD700"})
                first_linear = False
            layers_struct.append({"name": "Hidden", "size": layer.out_features, "color": "#00F0FF"})
            
    if len(layers_struct) > 1:
        layers_struct[-1]["name"] = "Output"
        layers_struct[-1]["color"] = "#7000FF"

    # 2. Calculate Coordinates
    # Max neurons to display per layer to keep it performant/readable
    MAX_NEURONS = 16 
    
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    layer_coords = [] # Store list of (x, y) for each layer to draw edges
    
    for layer_idx, info in enumerate(layers_struct):
        size = info['size']
        # Truncate if too big
        count = min(size, MAX_NEURONS)
        
        # Center the neurons along Y axis
        # range is centered around 0
        y_positions = [i - (count - 1)/2 for i in range(count)]
        
        current_layer_xy = []
        for y in y_positions:
            node_x.append(layer_idx)
            node_y.append(y)
            node_color.append(info['color'])
            txt = f"{info['name']}<br>Feature {int(y + (count-1)/2)}" if info['name'] == 'Input' else f"{info['name']}"
            if size > MAX_NEURONS and y == y_positions[-1]:
                 txt += "<br>(...)"
            node_text.append(txt)
            current_layer_xy.append((layer_idx, y))
            
        layer_coords.append(current_layer_xy)

    # 3. Create Edges
    edge_x = []
    edge_y = []
    
    for i in range(len(layer_coords) - 1):
        curr_layer = layer_coords[i]
        next_layer = layer_coords[i+1]
        
        for (x1, y1) in curr_layer:
            for (x2, y2) in next_layer:
                edge_x.extend([x1, x2, None])
                edge_y.extend([y1, y2, None])

    # 4. Plotly Traces
    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.6)', width=1), # White edges as requested
        hoverinfo='none',
        showlegend=False
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=25,
            color=node_color,
            line=dict(color='white', width=2),
            symbol='circle'
        ),
        text=[info['size'] if i == 0 else "" for i, x in enumerate(node_x)], # Show size only on first node of layer? No, simplified
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    # Annotations for Layer Names (Top of chart)
    annotations = []
    max_y = max([max(l_y) for l_y in [list(zip(*l))[1] for l in layer_coords]]) + 1
    
    for i, info in enumerate(layers_struct):
        annotations.append(dict(
            x=i, y=max_y,
            text=f"<b>{info['name']}</b><br>({info['size']})",
            showarrow=False,
            font=dict(color='white', size=14)
        ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
        dragmode='pan', # Enable navigation
        annotations=annotations
    )
    
    # Enable zoom
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False
    }

    st.plotly_chart(fig, use_container_width=True, config=config)

def render_metrics(metrics_history: list, is_classification: bool = True):
    """
    Renders charts with Purple/Cyan theme.
    """
    if not metrics_history:
        return
        
    epochs = [m['epoch'] for m in metrics_history]
    
    # Colors: Train=Cyan (#00F0FF), Val=Purple (#7000FF)
    color_map = ["#00F0FF", "#7000FF"]
    
    st.markdown("### Historique")
    
    # 1. LOSS
    loss_data = pd.DataFrame({
        'Train Loss': [m['train_loss'] for m in metrics_history],
        'Val Loss': [m['val_loss'] for m in metrics_history],
        'epoch': epochs
    }).set_index('epoch')
    
    st.caption("Loss (Erreur)")
    st.line_chart(loss_data, color=color_map)

    # 2. Performance
    if is_classification:
        st.caption("Accuracy (Précision)")
        acc_data = pd.DataFrame({
            'Train Acc': [m['train_acc'] for m in metrics_history],
            'Val Acc': [m['val_acc'] for m in metrics_history],
            'epoch': epochs
        }).set_index('epoch')
        st.line_chart(acc_data, color=color_map)
    
    else:
        # REGRESSION
        c1, c2 = st.columns(2)
        
        with c1:
            st.caption("Score R² (Fiabilité)")
            r2_data = pd.DataFrame({
                'Train R²': [m.get('train_r2', 0) for m in metrics_history],
                'Val R²': [m.get('val_r2', 0) for m in metrics_history],
                'epoch': epochs
            }).set_index('epoch')
            st.line_chart(r2_data, color=color_map)

        with c2:
            st.caption("RMSE (Erreur Moyenne)")
            rmse_data = pd.DataFrame({
                'Train RMSE': [m.get('train_rmse', 0) for m in metrics_history],
                'Val RMSE': [m.get('val_rmse', 0) for m in metrics_history],
                'epoch': epochs
            }).set_index('epoch')
            st.line_chart(rmse_data, color=color_map)

def render_evaluation_report(y_true, y_pred, is_classification: bool, class_names=None):
    """
    Renders detailed evaluation report.
    """
    if is_classification:
        st.markdown("### Matrice de Confusion")
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Purples')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown("### Prédictions vs Réalité")
        df_res = pd.DataFrame({'Target': y_true, 'Predicted': y_pred})
        
        fig = px.scatter(df_res, x='Target', y='Predicted', opacity=0.7, color_discrete_sequence=['#00F0FF'])
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig.add_shape(
            type="line", line=dict(dash='dash', color='#7000FF'),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='#333'),
            yaxis=dict(showgrid=True, gridcolor='#333')
        )
        
        st.plotly_chart(fig, use_container_width=True)
