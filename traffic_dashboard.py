# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 03:50:33 2024

@author: Adeka
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set the page configuration
st.set_page_config(page_title="PEMS04 Traffic Dashboard", layout="wide")

# Title
st.title("PEMS04 Traffic Dashboard")

# Sidebar for file uploads
st.sidebar.header("Upload Traffic Data")

# Function to load .npz files
@st.cache(allow_output_mutation=True)
def load_npz(file):
    data = np.load(file, allow_pickle=True)
    return data['data']

# Upload Actual Data
actual_uploaded_file = st.sidebar.file_uploader("Upload Actual Traffic Data (.npz)", type="npz")

# Upload Predicted Data
predicted_uploaded_file = st.sidebar.file_uploader("Upload Predicted Traffic Data (.npz)", type="npz")

# Initialize placeholders
actual_data = None
predicted_data = None

# Load Actual Data
if actual_uploaded_file is not None:
    try:
        actual_data = load_npz(actual_uploaded_file)
        st.sidebar.success("Actual data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading actual data: {e}")

# Load Predicted Data
if predicted_uploaded_file is not None:
    try:
        predicted_data = load_npz(predicted_uploaded_file)
        st.sidebar.success("Predicted data loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading predicted data: {e}")

# Check if both actual and predicted data are loaded
if actual_data is not None and predicted_data is not None:
    # Ensure that both datasets have the same shape
    if actual_data.shape != predicted_data.shape:
        st.error("Actual and Predicted data must have the same shape (timesteps, nodes, features).")
    else:
        # Extract dimensions
        timesteps, num_nodes, num_features = actual_data.shape

        # Sidebar selectors
        st.sidebar.header("Visualization Options")

        # Feature selection
        feature_options = ["Flow", "Occupancy", "Speed"]
        selected_feature = st.sidebar.selectbox("Select Feature", feature_options)

        # Map feature names to indices
        feature_map = {
            "Flow": 0,
            "Occupancy": 1,
            "Speed": 2
        }
        feature_idx = feature_map[selected_feature]

        # Node selection
        node_options = list(range(num_nodes))  # Assuming nodes are indexed from 0 to 306
        selected_node = st.sidebar.selectbox("Select Node (Sensor)", node_options)

        # Main Content Area
        st.header(f"Actual vs Predicted {selected_feature} for Node {selected_node}")

        # Extract time-series data for the selected node and feature
        actual_series = actual_data[:, selected_node, feature_idx]
        predicted_series = predicted_data[:, selected_node, feature_idx]

        # Create a DataFrame for plotting
        df = pd.DataFrame({
            "Timestep": np.arange(timesteps),
            "Actual": actual_series,
            "Predicted": predicted_series
        })

        # Plot using Plotly for interactivity
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Timestep"],
            y=df["Actual"],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=df["Timestep"],
            y=df["Predicted"],
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"Actual vs Predicted {selected_feature} for Node {selected_node}",
            xaxis_title="Timestep",
            yaxis_title=selected_feature,
            legend=dict(x=0, y=1),
            hovermode='x unified',
            width=1000,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Additional Statistics
        st.subheader("Statistics")

        actual_min = np.min(actual_series)
        actual_max = np.max(actual_series)
        predicted_min = np.min(predicted_series)
        predicted_max = np.max(predicted_series)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Actual {selected_feature}:**")
            st.write(f"Minimum: {actual_min:.2f}")
            st.write(f"Maximum: {actual_max:.2f}")

        with col2:
            st.write(f"**Predicted {selected_feature}:**")
            st.write(f"Minimum: {predicted_min:.2f}")
            st.write(f"Maximum: {predicted_max:.2f}")

        # Evaluation Metrics
        mae = mean_absolute_error(actual_series, predicted_series)
        rmse = np.sqrt(mean_squared_error(actual_series, predicted_series))

        st.write("**Evaluation Metrics:**")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Optional: Display Actual and Predicted Data in a Table
        st.subheader("Actual vs Predicted Data (Sample)")

        sample_size = 50  # Number of samples to display
        st.write(df.head(sample_size))

else:
    st.info("Please upload both Actual and Predicted traffic data files to visualize Actual vs. Predicted values.")

# --- Existing functionalities below ---
# Example: Data Overview, Timestep Selection, Histograms, Network Graph, etc.

# Data Overview
st.header("Data Overview")

if actual_data is not None:
    st.subheader("Actual Data Arrays")
    st.write(actual_uploaded_file.name)
    st.write(f"Shape: {actual_data.shape}")

if predicted_data is not None:
    st.subheader("Predicted Data Arrays")
    st.write(predicted_uploaded_file.name)
    st.write(f"Shape: {predicted_data.shape}")

# Timestep Selection
st.header("Select Timestep")

if actual_data is not None:
    timestep = st.slider("Select Timestep", min_value=0, max_value=actual_data.shape[0]-1, value=0)
    st.write(f"Selected Timestep: {timestep}")

    # Display statistics for selected timestep
    flow = actual_data[timestep, :, 0]
    occupancy = actual_data[timestep, :, 1]
    speed = actual_data[timestep, :, 2]

    st.subheader(f"Statistics at Timestep {timestep}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Flow**")
        st.write(f"Min: {flow.min():.2f}")
        st.write(f"Max: {flow.max():.2f}")

    with col2:
        st.write("**Occupancy (%)**")
        st.write(f"Min: {occupancy.min():.4f}")
        st.write(f"Max: {occupancy.max():.4f}")

    with col3:
        st.write("**Speed (km/h)**")
        st.write(f"Min: {speed.min():.2f}")
        st.write(f"Max: {speed.max():.2f}")

    # Histograms
    st.subheader(f"Traffic Metrics Distribution at Timestep {timestep}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(flow, bins=30, color='blue', alpha=0.7)
    axes[0].set_title("Flow Distribution")
    axes[0].set_xlabel("Flow")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(occupancy, bins=30, color='green', alpha=0.7)
    axes[1].set_title("Occupancy Distribution")
    axes[1].set_xlabel("Occupancy (%)")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(speed, bins=30, color='red', alpha=0.7)
    axes[2].set_title("Speed Distribution")
    axes[2].set_xlabel("Speed (km/h)")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    st.pyplot(fig)

# Network Graph
st.header("Network Graph Visualization")

if actual_data is not None:
    feature_options_graph = ["Flow", "Occupancy", "Speed"]
    selected_feature_graph = st.selectbox("Select Feature for Network Graph", feature_options_graph)

    # Map feature names to indices
    feature_map_graph = {
        "Flow": 0,
        "Occupancy": 1,
        "Speed": 2
    }
    feature_idx_graph = feature_map_graph[selected_feature_graph]

    # Extract data for the selected feature at the chosen timestep
    feature_data = actual_data[timestep, :, feature_idx_graph]

    # Construct Adjacency Matrix (Linear Topology)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if i > 0:
            adj_matrix[i][i-1] = 1
        if i < num_nodes - 1:
            adj_matrix[i][i+1] = 1

    # Create Network Graph
    G = nx.from_numpy_array(adj_matrix)

    # Assign feature values to nodes
    for node in G.nodes():
        G.nodes[node][selected_feature_graph.lower()] = feature_data[node]

    # Define node positions
    pos = nx.spring_layout(G, seed=42)  # For consistent layout

    # Create Plotly traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node][selected_feature_graph.lower()])
        node_text.append(f'Node {node}<br>{selected_feature_graph}: {G.nodes[node][selected_feature_graph.lower()]:.2f}')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title=selected_feature_graph,
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    fig_network = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=f'Network Graph - {selected_feature_graph} at Timestep {timestep}',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                annotations=[dict(
                                    text="",
                                    showarrow=False,
                                    xref="paper", yref="paper"
                                )],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )

    st.plotly_chart(fig_network, use_container_width=True)

# Optionally, add more features like Time Series Analysis, etc.

