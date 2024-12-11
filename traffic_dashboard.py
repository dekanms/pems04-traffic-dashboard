# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 01:18:49 2024

@author: Adeka
"""

# traffic_dashboard.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Function to load .npz data
@st.cache(allow_output_mutation=True)
def load_npz(file):
    data = np.load(file, allow_pickle=True)
    return data

# Function to plot histograms
def plot_histograms(flow, occupancy, speed):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(flow, bins=30, color='blue', edgecolor='black')
    axes[0].set_title('Flow Distribution')
    axes[0].set_xlabel('Flow')
    axes[0].set_ylabel('Number of Nodes')
    
    axes[1].hist(occupancy, bins=30, color='green', edgecolor='black')
    axes[1].set_title('Occupancy Distribution')
    axes[1].set_xlabel('Occupancy (%)')
    axes[1].set_ylabel('Number of Nodes')
    
    axes[2].hist(speed, bins=30, color='red', edgecolor='black')
    axes[2].set_title('Speed Distribution')
    axes[2].set_xlabel('Speed (km/h)')
    axes[2].set_ylabel('Number of Nodes')
    
    plt.tight_layout()
    st.pyplot(fig)

# Function to visualize network graph
def visualize_network_graph(adj_matrix, metric, feature_label='Flow'):
    G = nx.from_numpy_array(adj_matrix)
    
    # Assign metric values to nodes
    for node in G.nodes():
        G.nodes[node][feature_label.lower()] = metric[node]
    
    # Normalize metric for color mapping
    metrics = [G.nodes[node][feature_label.lower()] for node in G.nodes()]
    max_metric = max(metrics)
    min_metric = min(metrics)
    norm_metrics = [
        (m - min_metric) / (max_metric - min_metric) if max_metric != min_metric else 0.5
        for m in metrics
    ]
    
    # Define node colors
    cmap = plt.cm.viridis
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=50,
        node_color=norm_metrics,
        cmap=cmap,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    # Create colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=min_metric, vmax=max_metric)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(feature_label)
    
    plt.title(f'Network Graph Visualization - {feature_label}')
    plt.axis('off')
    st.pyplot(fig)

# Main Streamlit App
def main():
    st.title("PEMS04 Traffic Dashboard")
    
    # Sidebar for file upload
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a .npz file", type="npz")
    
    if uploaded_file is not None:
        # Load data
        data = load_npz(uploaded_file)
        st.success("File uploaded and loaded successfully!")
        
        # Display available arrays
        st.subheader("Data Overview")
        st.write("Available arrays in the .npz file:", data.files)
        
        # Extract the 'data' array
        data_array = data['data']
        st.write(f"Shape of 'data' array: {data_array.shape}")
        
        # Verify shape (timesteps, nodes, features)
        if data_array.ndim != 3 or data_array.shape[2] != 3:
            st.error("Unexpected data shape. Expected shape: (timesteps, nodes, 3)")
            return
        
        # Assuming features are ordered as [Flow, Occupancy, Speed]
        # Adjust indices if your data has a different order
        flow = data_array[:, :, 0]        # Shape: (timesteps, nodes)
        occupancy = data_array[:, :, 1]   # Shape: (timesteps, nodes)
        speed = data_array[:, :, 2]       # Shape: (timesteps, nodes)
        
        # Select a timestep to visualize
        timestep = st.slider(
            "Select Timestep for Visualization",
            min_value=0,
            max_value=data_array.shape[0]-1,
            value=0,
            format="Timestep %d"
        )
        st.write(f"Selected Timestep: {timestep}")
        
        # Display basic statistics for the selected timestep
        st.subheader("Traffic Metrics Statistics at Selected Timestep")
        st.write(f"Flow: Min = {flow[timestep].min():.2f}, Max = {flow[timestep].max():.2f}")
        st.write(f"Occupancy: Min = {occupancy[timestep].min():.4f}, Max = {occupancy[timestep].max():.4f}")
        st.write(f"Speed: Min = {speed[timestep].min():.2f}, Max = {speed[timestep].max():.2f}")
        
        # Generate Histograms
        st.subheader("Traffic Metrics Distributions")
        plot_histograms(flow[timestep], occupancy[timestep], speed[timestep])
        
        # Construct Adjacency Matrix
        # Assuming a linear topology: node i connected to node i-1 and i+1
        num_nodes = data_array.shape[1]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if i > 0:
                adj_matrix[i][i-1] = 1
            if i < num_nodes - 1:
                adj_matrix[i][i+1] = 1
        
        # Select feature to visualize on Network Graph
        feature = st.selectbox("Select Feature to Visualize on Network Graph", ["Flow", "Occupancy", "Speed"])
        feature_map = {
            "Flow": flow[timestep],
            "Occupancy": occupancy[timestep],
            "Speed": speed[timestep]
        }
        feature_label_map = {
            "Flow": "Flow",
            "Occupancy": "Occupancy (%)",
            "Speed": "Speed (km/h)"
        }
        selected_feature = feature_map[feature]
        selected_label = feature_label_map[feature]
        
        # Generate Network Graph
        st.subheader("Network Graph Visualization")
        visualize_network_graph(adj_matrix, selected_feature, feature_label=selected_label)

if __name__ == "__main__":
    main()
