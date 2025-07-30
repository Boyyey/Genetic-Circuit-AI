"""
Visualizer Module for Genetic Circuit Design Platform.

This module provides advanced visualization capabilities for genetic circuits,
including interactive plots, circuit diagrams, and analysis visualizations.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import networkx as nx
from models.circuit import Circuit, CircuitNode, CircuitEdge, NodeType, EdgeType
from models.simulation import SimulationResult, TimeSeries
from models.parts import GeneticPart


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    
    width: int = 800
    height: int = 600
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    color_scheme: str = "plotly"
    template: str = "plotly_white"
    show_legend: bool = True
    show_grid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "width": self.width,
            "height": self.height,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "color_scheme": self.color_scheme,
            "template": self.template,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid
        }


class CircuitVisualizer:
    """Advanced circuit visualizer with interactive plotting capabilities."""
    
    def __init__(self):
        """Initialize the circuit visualizer."""
        self.color_palettes = self._load_color_palettes()
        self.layout_templates = self._load_layout_templates()
        
        logger.info("Circuit visualizer initialized successfully")
    
    def _load_color_palettes(self) -> Dict[str, List[str]]:
        """Load color palettes for visualization."""
        return {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
            "biological": ["#2E8B57", "#FF6347", "#4169E1", "#FFD700", "#8A2BE2", "#FF69B4", "#00CED1", "#FF4500"],
            "circuit": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"],
            "scientific": ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#34495E"]
        }
    
    def _load_layout_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load layout templates for different plot types."""
        return {
            "circuit_diagram": {
                "title": "Genetic Circuit Diagram",
                "showlegend": True,
                "hovermode": "closest",
                "margin": dict(b=20, l=5, r=5, t=40),
                "plot_bgcolor": "white"
            },
            "time_series": {
                "title": "Gene Expression Over Time",
                "xaxis_title": "Time (hours)",
                "yaxis_title": "Expression Level",
                "hovermode": "x unified",
                "plot_bgcolor": "white"
            },
            "parameter_analysis": {
                "title": "Parameter Analysis",
                "xaxis_title": "Parameter Value",
                "yaxis_title": "Output",
                "hovermode": "closest",
                "plot_bgcolor": "white"
            }
        }
    
    def create_circuit_diagram(self, circuit: Circuit, config: Optional[PlotConfig] = None) -> go.Figure:
        """Create an interactive circuit diagram."""
        logger.info(f"Creating circuit diagram for circuit: {circuit.name}")
        
        if config is None:
            config = PlotConfig(
                title=f"Circuit Diagram: {circuit.name}",
                width=1000,
                height=700
            )
        
        # Create graph from circuit topology
        G = circuit.topology.to_networkx()
        
        # Generate layout
        pos = self._generate_layout(G)
        
        # Create node trace
        node_trace = self._create_node_trace(G, pos, config)
        
        # Create edge traces
        edge_traces = self._create_edge_traces(G, pos, config)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        # Update layout
        fig.update_layout(
            title=config.title,
            showlegend=config.show_legend,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def _generate_layout(self, G: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Generate layout for circuit diagram."""
        # Try different layout algorithms
        try:
            # Use spring layout for small graphs
            if len(G.nodes) <= 10:
                pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            else:
                # Use hierarchical layout for larger graphs
                pos = nx.kamada_kawai_layout(G)
        except Exception as e:
            logger.warning(f"Layout generation failed: {e}, using circular layout")
            pos = nx.circular_layout(G)
        
        return pos
    
    def _create_node_trace(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                          config: PlotConfig) -> go.Scatter:
        """Create node trace for circuit diagram."""
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Get node colors and sizes
        node_colors = []
        node_sizes = []
        node_text = []
        node_hover_text = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get("node_type", "gene")
            
            # Set color based on node type
            color = self._get_node_color(node_type)
            node_colors.append(color)
            
            # Set size based on node type
            size = self._get_node_size(node_type)
            node_sizes.append(size)
            
            # Set text
            node_text.append(node_data.get("name", node))
            
            # Set hover text
            hover_text = f"{node_data.get('name', node)}<br>Type: {node_type}"
            if "properties" in node_data:
                for key, value in node_data["properties"].items():
                    hover_text += f"<br>{key}: {value}"
            node_hover_text.append(hover_text)
        
        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color="white"),
                symbol="circle"
            ),
            textfont=dict(size=10, color="white"),
            hoverinfo="text",
            hovertext=node_hover_text,
            name="Components"
        )
    
    def _create_edge_traces(self, G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], 
                           config: PlotConfig) -> List[go.Scatter]:
        """Create edge traces for circuit diagram."""
        edge_traces = []
        
        # Group edges by type
        edge_types = {}
        for source, target, data in G.edges(data=True):
            edge_type = data.get("edge_type", "regulation")
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            edge_types[edge_type].append((source, target, data))
        
        # Create trace for each edge type
        for edge_type, edges in edge_types.items():
            edge_x = []
            edge_y = []
            edge_hover_text = []
            
            for source, target, data in edges:
                # Add source position
                edge_x.append(pos[source][0])
                edge_y.append(pos[source][1])
                
                # Add target position
                edge_x.append(pos[target][0])
                edge_y.append(pos[target][1])
                
                # Add None for line break
                edge_x.append(None)
                edge_y.append(None)
                
                # Add hover text
                hover_text = f"{source} → {target}<br>Type: {edge_type}"
                if "weight" in data:
                    hover_text += f"<br>Weight: {data['weight']}"
                edge_hover_text.append(hover_text)
            
            # Create trace
            trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=2,
                    color=self._get_edge_color(edge_type)
                ),
                hoverinfo="text",
                hovertext=edge_hover_text,
                name=edge_type.replace("_", " ").title(),
                showlegend=True
            )
            edge_traces.append(trace)
        
        return edge_traces
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        colors = {
            "promoter": "#FF6B6B",
            "gene": "#4ECDC4",
            "protein": "#45B7D1",
            "terminator": "#96CEB4",
            "ribosome_binding_site": "#FFEAA7",
            "small_molecule": "#DDA0DD",
            "logic_gate": "#98D8C8",
            "sensor": "#F7DC6F",
            "reporter": "#BB8FCE",
            "regulator": "#85C1E9"
        }
        return colors.get(node_type, "#BDC3C7")
    
    def _get_node_size(self, node_type: str) -> int:
        """Get size for node type."""
        sizes = {
            "promoter": 25,
            "gene": 30,
            "protein": 28,
            "terminator": 20,
            "ribosome_binding_site": 22,
            "small_molecule": 24,
            "logic_gate": 26,
            "sensor": 27,
            "reporter": 29,
            "regulator": 31
        }
        return sizes.get(node_type, 25)
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for edge type."""
        colors = {
            "transcription": "#E74C3C",
            "translation": "#3498DB",
            "regulation": "#2ECC71",
            "degradation": "#F39C12",
            "binding": "#9B59B6",
            "catalysis": "#1ABC9C",
            "inhibition": "#E67E22",
            "activation": "#34495E"
        }
        return colors.get(edge_type, "#95A5A6")
    
    def create_time_series_plot(self, time_series: TimeSeries, config: Optional[PlotConfig] = None) -> go.Figure:
        """Create time series plot."""
        logger.info("Creating time series plot")
        
        if config is None:
            config = PlotConfig(
                title="Gene Expression Over Time",
                x_label="Time (hours)",
                y_label="Expression Level",
                width=900,
                height=600
            )
        
        fig = go.Figure()
        
        # Add traces for each variable
        colors = self.color_palettes["biological"]
        for i, (variable, values) in enumerate(time_series.values.items()):
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=time_series.time_points,
                y=values,
                mode="lines",
                name=variable,
                line=dict(width=2, color=color),
                hovertemplate=f"{variable}<br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            hovermode="x unified",
            plot_bgcolor="white",
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_parameter_analysis_plot(self, parameter_values: List[float], 
                                     output_values: List[float], 
                                     parameter_name: str,
                                     config: Optional[PlotConfig] = None) -> go.Figure:
        """Create parameter analysis plot."""
        logger.info(f"Creating parameter analysis plot for {parameter_name}")
        
        if config is None:
            config = PlotConfig(
                title=f"Parameter Analysis: {parameter_name}",
                x_label=f"{parameter_name} Value",
                y_label="Output",
                width=800,
                height=500
            )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=parameter_values,
            y=output_values,
            mode="lines+markers",
            name="Output",
            line=dict(width=2, color="#3498DB"),
            marker=dict(size=6, color="#3498DB"),
            hovertemplate=f"{parameter_name}: %{{x}}<br>Output: %{{y}}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            plot_bgcolor="white",
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_sensitivity_plot(self, sensitivities: Dict[str, float], 
                              config: Optional[PlotConfig] = None) -> go.Figure:
        """Create sensitivity analysis plot."""
        logger.info("Creating sensitivity analysis plot")
        
        if config is None:
            config = PlotConfig(
                title="Parameter Sensitivity Analysis",
                x_label="Parameters",
                y_label="Sensitivity",
                width=800,
                height=500
            )
        
        # Sort parameters by sensitivity
        sorted_sensitivities = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
        parameters = [item[0] for item in sorted_sensitivities]
        sensitivity_values = [abs(item[1]) for item in sorted_sensitivities]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=parameters,
            y=sensitivity_values,
            marker_color="#E74C3C",
            hovertemplate="Parameter: %{x}<br>Sensitivity: %{y}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            plot_bgcolor="white",
            width=config.width,
            height=config.height,
            showlegend=False
        )
        
        return fig
    
    def create_comparison_plot(self, results: List[SimulationResult], 
                             config: Optional[PlotConfig] = None) -> go.Figure:
        """Create comparison plot for multiple simulation results."""
        logger.info("Creating comparison plot")
        
        if config is None:
            config = PlotConfig(
                title="Circuit Comparison",
                x_label="Time (hours)",
                y_label="Expression Level",
                width=1000,
                height=600
            )
        
        fig = go.Figure()
        
        colors = self.color_palettes["circuit"]
        for i, result in enumerate(results):
            circuit_name = result.metadata.get("circuit_name", f"Circuit {i+1}")
            color = colors[i % len(colors)]
            
            # Add traces for each variable
            for variable, values in result.time_series.values.items():
                fig.add_trace(go.Scatter(
                    x=result.time_series.time_points,
                    y=values,
                    mode="lines",
                    name=f"{circuit_name} - {variable}",
                    line=dict(width=2, color=color),
                    hovertemplate=f"{circuit_name} - {variable}<br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            hovermode="x unified",
            plot_bgcolor="white",
            width=config.width,
            height=config.height,
            showlegend=config.show_legend
        )
        
        return fig
    
    def create_heatmap(self, data: np.ndarray, row_labels: List[str], 
                      col_labels: List[str], config: Optional[PlotConfig] = None) -> go.Figure:
        """Create heatmap plot."""
        logger.info("Creating heatmap plot")
        
        if config is None:
            config = PlotConfig(
                title="Parameter Heatmap",
                width=800,
                height=600
            )
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=col_labels,
            y=row_labels,
            colorscale="Viridis",
            hovertemplate="Row: %{y}<br>Column: %{x}<br>Value: %{z}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            plot_bgcolor="white",
            width=config.width,
            height=config.height
        )
        
        return fig
    
    def create_3d_surface(self, x_values: np.ndarray, y_values: np.ndarray, 
                         z_values: np.ndarray, config: Optional[PlotConfig] = None) -> go.Figure:
        """Create 3D surface plot."""
        logger.info("Creating 3D surface plot")
        
        if config is None:
            config = PlotConfig(
                title="3D Parameter Surface",
                width=900,
                height=700
            )
        
        fig = go.Figure(data=go.Surface(
            x=x_values,
            y=y_values,
            z=z_values,
            colorscale="Viridis",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=config.title,
            scene=dict(
                xaxis_title="Parameter 1",
                yaxis_title="Parameter 2",
                zaxis_title="Output"
            ),
            width=config.width,
            height=config.height
        )
        
        return fig


class PlotGenerator:
    """Specialized plot generator for different analysis types."""
    
    def __init__(self):
        """Initialize the plot generator."""
        self.visualizer = CircuitVisualizer()
    
    def generate_analysis_dashboard(self, circuit: Circuit, 
                                  simulation_result: SimulationResult) -> go.Figure:
        """Generate a comprehensive analysis dashboard."""
        logger.info("Generating analysis dashboard")
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=("Circuit Diagram", "Time Series", "Parameter Analysis", "Sensitivity"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add circuit diagram
        circuit_fig = self.visualizer.create_circuit_diagram(circuit)
        for trace in circuit_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add time series
        time_series_fig = self.visualizer.create_time_series_plot(simulation_result.time_series)
        for trace in time_series_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title="Genetic Circuit Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_report_plots(self, circuit: Circuit, 
                            simulation_result: SimulationResult) -> Dict[str, go.Figure]:
        """Generate all plots for a comprehensive report."""
        logger.info("Generating report plots")
        
        plots = {}
        
        # Circuit diagram
        plots["circuit_diagram"] = self.visualizer.create_circuit_diagram(circuit)
        
        # Time series
        plots["time_series"] = self.visualizer.create_time_series_plot(simulation_result.time_series)
        
        # Performance metrics
        metrics = simulation_result.calculate_metrics()
        if metrics:
            plots["performance_metrics"] = self._create_metrics_plot(metrics)
        
        # Steady state analysis
        steady_state = simulation_result.get_steady_state()
        if steady_state:
            plots["steady_state"] = self._create_steady_state_plot(steady_state)
        
        return plots
    
    def _create_metrics_plot(self, metrics: Dict[str, float]) -> go.Figure:
        """Create performance metrics plot."""
        fig = go.Figure()
        
        # Create bar chart of metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color="#3498DB"
        ))
        
        fig.update_layout(
            title="Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            plot_bgcolor="white"
        )
        
        return fig
    
    def _create_steady_state_plot(self, steady_state: Dict[str, float]) -> go.Figure:
        """Create steady state analysis plot."""
        fig = go.Figure()
        
        variables = list(steady_state.keys())
        values = list(steady_state.values())
        
        fig.add_trace(go.Bar(
            x=variables,
            y=values,
            marker_color="#2ECC71"
        ))
        
        fig.update_layout(
            title="Steady State Analysis",
            xaxis_title="Variable",
            yaxis_title="Steady State Value",
            plot_bgcolor="white"
        )
        
        return fig 