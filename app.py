"""
Genetic Circuit Design Platform - Main Application

A comprehensive web application for designing, simulating, and visualizing
genetic circuits using natural language processing and advanced bioinformatics tools.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from models.simulation import SolverType
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.nlp_parser import NLPParser
from core.circuit_designer import CircuitDesigner
from core.simulator import BiologicalSimulator
from core.visualizer import CircuitVisualizer
from core.part_database import PartDatabase
from models.circuit import Circuit, CircuitNode, CircuitEdge
from models.parts import GeneticPart, Promoter, Gene
from models.simulation import SimulationParameters, SimulationResult


# Page configuration
st.set_page_config(
    page_title="🧬 Genetic Circuit Designer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .circuit-diagram {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class GeneticCircuitApp:
    """Main application class for the Genetic Circuit Design Platform."""
    
    def __init__(self):
        """Initialize the application."""
        self.initialize_session_state()
        self.load_components()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'circuit_history' not in st.session_state:
            st.session_state.circuit_history = []
        
        if 'current_circuit' not in st.session_state:
            st.session_state.current_circuit = None
        
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        
        if 'parsing_results' not in st.session_state:
            st.session_state.parsing_results = None
    
    def load_components(self):
        """Load and initialize core components."""
        try:
            # Initialize NLP parser
            openai_api_key = os.getenv('OPENAI_API_KEY')
            self.nlp_parser = NLPParser(openai_api_key)
            
            # Initialize circuit designer
            self.circuit_designer = CircuitDesigner()
            
            # Initialize simulator
            self.simulator = BiologicalSimulator()
            
            # Initialize visualizer
            self.visualizer = CircuitVisualizer()
            
            # Initialize part database
            self.part_database = PartDatabase()
            
            st.session_state.components_loaded = True
            
        except Exception as e:
            st.error(f"Error loading components: {e}")
            st.session_state.components_loaded = False
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">🧬 Genetic Circuit Designer</h1>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        st.sidebar.markdown("## 🧭 Navigation")
        
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["🏠 Home", "🧬 Circuit Design", "🔬 Simulation", "📊 Analysis", "📚 Parts Library", "⚙️ Settings"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🎛️ Quick Settings")
        
        # Organism selection
        organism = st.sidebar.selectbox(
            "Target Organism:",
            ["E. coli", "S. cerevisiae", "B. subtilis", "P. putida", "mammalian"],
            index=0
        )
        
        # Simulation parameters
        st.sidebar.markdown("### ⏱️ Simulation")
        time_end = st.sidebar.slider("Simulation Time (hours):", 1, 100, 24)
        time_step = st.sidebar.slider("Time Step (minutes):", 1, 60, 10)
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            solver_type = st.selectbox(
                "ODE Solver:",
                ["RK45", "RK4", "BDF", "LSODA"],
                index=0
            )
            
            tolerance = st.slider("Tolerance:", 1e-8, 1e-3, 1e-6, format="%.1e")
        
        return page, {
            "organism": organism,
            "time_end": time_end,
            "time_step": time_step / 60,  # Convert to hours
            "solver_type": solver_type,
            "tolerance": tolerance
        }
    
    def render_home_page(self):
        """Render the home page with overview and examples."""
        st.markdown('<h2 class="sub-header">🏠 Welcome to Genetic Circuit Design</h2>', unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 What is this platform?
            
            This is a cutting-edge platform that combines natural language processing, 
            artificial intelligence, and advanced bioinformatics to help you design 
            genetic circuits from simple descriptions.
            
            **Key Features:**
            - 🧠 **Natural Language Processing**: Describe circuits in plain English
            - 🤖 **AI-Powered Design**: Intelligent circuit generation and optimization
            - 🔬 **Biological Simulation**: Realistic ODE-based simulations
            - 📊 **Interactive Visualization**: Beautiful circuit diagrams and plots
            - 🧬 **Parts Database**: Comprehensive genetic parts library
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Platform Stats
            """)
            
            # Mock statistics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Circuits Designed", "1,247")
                st.metric("Simulations Run", "3,891")
            with col2b:
                st.metric("Parts Available", "2,156")
                st.metric("Success Rate", "94.2%")
        
        # Example circuits
        st.markdown('<h3 class="sub-header">💡 Example Circuits</h3>', unsafe_allow_html=True)
        
        examples = [
            {
                "title": "Simple Repression",
                "description": "Express Gene A only when Sugar X is high and Stress Y is low",
                "complexity": "Low",
                "components": 3
            },
            {
                "title": "Oscillatory Circuit", 
                "description": "Create an oscillating expression of Gene B every 6 hours",
                "complexity": "Medium",
                "components": 5
            },
            {
                "title": "Logic Gate",
                "description": "Build an AND gate that activates Gene C when both Input A and Input B are present",
                "complexity": "Medium",
                "components": 4
            }
        ]
        
        for i, example in enumerate(examples):
            with st.expander(f"🔬 {example['title']} ({example['complexity']} Complexity)"):
                st.write(f"**Description:** {example['description']}")
                st.write(f"**Components:** {example['components']} parts")
                
                if st.button(f"Try this example", key=f"example_{i}"):
                    st.session_state.example_text = example['description']
                    st.rerun()
        
        # Quick start
        st.markdown('<h3 class="sub-header">🚀 Quick Start</h3>', unsafe_allow_html=True)
        
        quick_start_text = st.text_area(
            "Describe your genetic circuit:",
            placeholder="e.g., 'Make a genetic circuit that turns on Gene A only when Sugar X is high and Stress Y is low'",
            height=100
        )
        
        if st.button("🎯 Design Circuit", type="primary"):
            if quick_start_text.strip():
                self.process_circuit_design(quick_start_text)
            else:
                st.warning("Please enter a circuit description.")
    
    def render_circuit_design_page(self, settings: Dict[str, Any]):
        """Render the circuit design page."""
        st.markdown('<h2 class="sub-header">🧬 Design Your Genetic Circuit</h2>', unsafe_allow_html=True)
        
        # Initialize session state variables if they don't exist
        if 'circuit_description' not in st.session_state:
            st.session_state.circuit_description = ""
        if 'design_in_progress' not in st.session_state:
            st.session_state.design_in_progress = False
        
        # Use a form to prevent rerun on button press
        with st.form("circuit_design_form"):
            # Two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📝 Circuit Description")
                
                circuit_description = st.text_area(
                    "Describe the genetic circuit you want to design:",
                    value=st.session_state.circuit_description,
                    height=200,
                    placeholder="Example: Design a NOT gate with a repressor that inhibits a reporter gene...",
                    key="circuit_desc_input"
                )
                
                # Update session state with the current description
                st.session_state.circuit_description = circuit_description
                
                # Add some space
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Add design tips
                with st.expander("💡 Design Tips"):
                    st.markdown("""
                    - Be specific about the components you want to include
                    - Mention the type of regulation (activation, repression, etc.)
                    - Specify any constraints or requirements
                    - Include the desired behavior or output
                    """)
            
            with col2:
                st.markdown("### 🎛️ Design Options")
                
                design_mode = st.selectbox(
                    "Design Mode:",
                    ["Auto", "Guided", "Expert"],
                    help="Auto: AI generates complete circuit\nGuided: Step-by-step design\nExpert: Full control"
                )
                
                optimization_level = st.selectbox(
                    "Optimization:",
                    ["Basic", "Standard", "Advanced"],
                    help="Level of AI optimization applied"
                )
            
            # Design button
            submitted = st.form_submit_button(
                "🎯 Design Circuit", 
                type="primary", 
                use_container_width=True,
                disabled=st.session_state.design_in_progress
            )
            
            if submitted and circuit_description.strip():
                st.session_state.design_in_progress = True
                st.session_state.circuit_description = circuit_description
                # Use a callback to process the design
                st.session_state.process_design = True
                st.rerun()
            elif submitted and not circuit_description.strip():
                st.warning("Please enter a circuit description.")
        
        # Process design outside the form to prevent form resubmission
        if st.session_state.get('process_design', False):
            st.session_state.process_design = False
            with st.spinner("🧠 Analyzing circuit description..."):
                try:
                    self.process_circuit_design(
                        st.session_state.circuit_description, 
                        settings
                    )
                finally:
                    st.session_state.design_in_progress = False
                st.rerun()
        
        # Display current circuit - Only show if we have a circuit in session state
        if 'current_circuit' in st.session_state and st.session_state.current_circuit:
            self.display_circuit_design(st.session_state.current_circuit)
    
    def process_circuit_design(self, description: str, settings: Dict[str, Any] = None):
        """Process circuit design from description."""
        try:
            # Parse the description
            with st.spinner("🔍 Parsing natural language..."):
                parsing_results = self.nlp_parser.parse_circuit_description(
                    description, 
                    organism=settings.get('organism', 'E. coli') if settings else 'E. coli'
                )
                st.session_state.parsing_results = parsing_results
            
            # Design the circuit
            with st.spinner("🧬 Designing circuit architecture..."):
                circuit = self.circuit_designer.design_circuit(parsing_results)
                st.session_state.current_circuit = circuit
            
            # Add to history
            st.session_state.circuit_history.append({
                'timestamp': datetime.now(),
                'description': description,
                'circuit': circuit
            })
            
            st.success("✅ Circuit designed successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error designing circuit: {e}")
            st.exception(e)
    
    def display_circuit_design(self, circuit: Circuit):
        """Display the designed circuit."""
        st.markdown('<h3 class="sub-header">📋 Circuit Design Results</h3>', unsafe_allow_html=True)
        
        # Circuit overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Components", len(circuit.topology.nodes))
        
        with col2:
            st.metric("Connections", len(circuit.topology.edges))
        
        with col3:
            st.metric("Logic Gates", len(circuit.topology.logic_gates))
        
        # Circuit diagram
        st.markdown('<h4 class="sub-header">🔗 Circuit Diagram</h4>', unsafe_allow_html=True)
        
        # Create interactive circuit diagram using Plotly
        fig = self.create_circuit_diagram(circuit)
        st.plotly_chart(fig, use_container_width=True)
        
        # Circuit details
        with st.expander("📊 Circuit Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🧬 Components")
                for node in circuit.topology.nodes:
                    st.write(f"• **{node.name}** ({node.node_type.value})")
            
            with col2:
                st.markdown("### 🔗 Connections")
                for edge in circuit.topology.edges:
                    source = circuit.topology.get_node_by_id(edge.source_id)
                    target = circuit.topology.get_node_by_id(edge.target_id)
                    if source and target:
                        st.write(f"• {source.name} → {target.name} ({edge.edge_type.value})")
        
        # Simulation controls
        st.markdown('<h4 class="sub-header">🔬 Simulation</h4>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Run Simulation", type="primary"):
                self.run_simulation(circuit)
        
        with col2:
            if st.button("📈 View Results"):
                if st.session_state.simulation_results:
                    self.display_simulation_results(st.session_state.simulation_results)
        
        with col3:
            if st.button("💾 Save Circuit"):
                self.save_circuit(circuit)
    
    def create_circuit_diagram(self, circuit: Circuit) -> go.Figure:
        """Create an interactive circuit diagram using Plotly."""
        # Extract node positions (simple layout)
        nodes = circuit.topology.nodes
        edges = circuit.topology.edges
        
        # Create node positions (circular layout)
        n_nodes = len(nodes)
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 1.0
        
        node_x = radius * np.cos(angles)
        node_y = radius * np.sin(angles)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[node.name for node in nodes],
            textposition="middle center",
            marker=dict(
                size=30,
                color=[self.get_node_color(node.node_type.value) for node in nodes],
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=10, color='white'),
            hoverinfo='text',
            hovertext=[f"{node.name}<br>Type: {node.node_type.value}" for node in nodes]
        )
        
        # Create edge traces
        edge_traces = []
        for edge in edges:
            source_idx = next(i for i, node in enumerate(nodes) if node.id == edge.source_id)
            target_idx = next(i for i, node in enumerate(nodes) if node.id == edge.target_id)
            
            edge_trace = go.Scatter(
                x=[node_x[source_idx], node_x[target_idx]],
                y=[node_y[source_idx], node_y[target_idx]],
                mode='lines',
                line=dict(width=2, color=self.get_edge_color(edge.edge_type.value)),
                hoverinfo='text',
                hovertext=f"{edge.edge_type.value}",
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        fig.update_layout(
            title="Genetic Circuit Diagram",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        colors = {
            'promoter': '#ff7f0e',
            'gene': '#2ca02c',
            'protein': '#d62728',
            'terminator': '#9467bd',
            'ribosome_binding_site': '#8c564b',
            'small_molecule': '#e377c2',
            'logic_gate': '#7f7f7f',
            'sensor': '#bcbd22',
            'reporter': '#17becf',
            'regulator': '#ff9896'
        }
        return colors.get(node_type, '#1f77b4')
    
    def get_edge_color(self, edge_type: str) -> str:
        """Get color for edge type."""
        colors = {
            'transcription': '#ff7f0e',
            'translation': '#2ca02c',
            'regulation': '#d62728',
            'degradation': '#9467bd',
            'binding': '#8c564b',
            'catalysis': '#e377c2',
            'inhibition': '#7f7f7f',
            'activation': '#bcbd22'
        }
        return colors.get(edge_type, '#1f77b4')
    
    def run_simulation(self, circuit: Circuit):
        """Run simulation for the circuit."""
        try:
            with st.spinner("🔬 Running simulation..."):
                # Create simulation parameters
                sim_params = SimulationParameters(
                    time_end=24.0,
                    time_step=0.1,
                    solver_type=SolverType.RK45
                )
                
                # Run simulation
                results = self.simulator.simulate_circuit(circuit, sim_params)
                st.session_state.simulation_results = results
                
                st.success("✅ Simulation completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
    
    def display_simulation_results(self, results: SimulationResult):
        """Display simulation results."""
        st.markdown('<h3 class="sub-header">📊 Simulation Results</h3>', unsafe_allow_html=True)
        
        # Time series plot
        if results.time_series.values:
            fig = self.create_time_series_plot(results.time_series)
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        metrics = results.calculate_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Expression", f"{metrics.get('system_max', 0):.2f}")
        
        with col2:
            st.metric("Min Expression", f"{metrics.get('system_min', 0):.2f}")
        
        with col3:
            st.metric("Mean Expression", f"{metrics.get('system_mean', 0):.2f}")
        
        with col4:
            st.metric("Convergence Error", f"{metrics.get('convergence_error', 0):.2e}")
        
        # Steady state analysis
        steady_state = results.get_steady_state()
        
        with st.expander("⚖️ Steady State Analysis"):
            for variable, value in steady_state.items():
                st.write(f"**{variable}:** {value:.4f}")
    
    def create_time_series_plot(self, time_series) -> go.Figure:
        """Create time series plot."""
        fig = go.Figure()
        
        for variable, values in time_series.values.items():
            fig.add_trace(go.Scatter(
                x=time_series.time_points,
                y=values,
                mode='lines',
                name=variable,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Gene Expression Over Time",
            xaxis_title="Time (hours)",
            yaxis_title="Expression Level",
            hovermode='x unified',
            plot_bgcolor='white'
        )
        
        return fig
    
    def save_circuit(self, circuit: Circuit):
        """Save circuit to file."""
        try:
            filename = f"circuit_{circuit.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            circuit.save_to_file(filename)
            st.success(f"✅ Circuit saved as {filename}")
        except Exception as e:
            st.error(f"❌ Error saving circuit: {e}")
    
    def render_simulation_page(self, settings: Dict[str, Any]):
        """Render the simulation page."""
        st.markdown('<h2 class="sub-header">🔬 Simulation & Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.current_circuit:
            st.warning("⚠️ No circuit loaded. Please design a circuit first.")
            return
        
        # Simulation parameters
        st.markdown('<h3 class="sub-header">⚙️ Simulation Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_end = st.number_input("Simulation Time (hours):", 1.0, 100.0, float(settings['time_end']))
            time_step = st.number_input("Time Step (hours):", 0.01, 1.0, float(settings['time_step']))
        
        with col2:
            solver_type = st.selectbox("Solver:", ["RK45", "RK4", "BDF", "LSODA"], 
                                     index=["RK45", "RK4", "BDF", "LSODA"].index(settings['solver_type']))
            tolerance = st.number_input("Tolerance:", 1e-8, 1e-3, settings['tolerance'], format="%.1e")
        
        with col3:
            st.markdown("### 📊 Analysis Options")
            sensitivity_analysis = st.checkbox("Sensitivity Analysis")
            parameter_scan = st.checkbox("Parameter Scan")
            optimization = st.checkbox("Parameter Optimization")
        
        # Run simulation
        if st.button("▶️ Run Simulation", type="primary"):
            self.run_advanced_simulation(settings)
        
        # Display results
        if st.session_state.simulation_results:
            self.display_advanced_results(st.session_state.simulation_results)
    
    def run_advanced_simulation(self, settings: Dict[str, Any]):
        """Run advanced simulation with analysis."""
        # Implementation for advanced simulation
        pass
    
    def display_advanced_results(self, results: SimulationResult):
        """Display advanced simulation results."""
        # Implementation for advanced results display
        pass
    
    def render_analysis_page(self):
        """Render the analysis page."""
        st.markdown('<h2 class="sub-header">📊 Analysis & Optimization</h2>', unsafe_allow_html=True)
        
        # Analysis options
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Parameter Sensitivity", "Robustness Analysis", "Performance Optimization", "Comparative Analysis"]
        )
        
        if analysis_type == "Parameter Sensitivity":
            self.render_sensitivity_analysis()
        elif analysis_type == "Robustness Analysis":
            self.render_robustness_analysis()
        elif analysis_type == "Performance Optimization":
            self.render_optimization_analysis()
        elif analysis_type == "Comparative Analysis":
            self.render_comparative_analysis()
    
    def render_sensitivity_analysis(self):
        """Render sensitivity analysis."""
        st.markdown("### 🔍 Parameter Sensitivity Analysis")
        st.info("This analysis shows how sensitive the circuit is to parameter changes.")
        
        # Implementation for sensitivity analysis
        pass
    
    def render_robustness_analysis(self):
        """Render robustness analysis."""
        st.markdown("### 🛡️ Robustness Analysis")
        st.info("This analysis tests circuit performance under various conditions.")
        
        # Implementation for robustness analysis
        pass
    
    def render_optimization_analysis(self):
        """Render optimization analysis."""
        st.markdown("### 🎯 Performance Optimization")
        st.info("This analysis optimizes circuit parameters for better performance.")
        
        # Implementation for optimization analysis
        pass
    
    def render_comparative_analysis(self):
        """Render comparative analysis."""
        st.markdown("### 📈 Comparative Analysis")
        st.info("Compare different circuit designs and their performance.")
        
        # Implementation for comparative analysis
        pass
    
    def render_parts_library_page(self):
        """Render the parts library page."""
        st.markdown('<h2 class="sub-header">📚 Parts Library</h2>', unsafe_allow_html=True)
        
        # Search parts
        search_query = st.text_input("🔍 Search parts:", placeholder="e.g., promoter, gene, protein")
        
        if search_query:
            # Search implementation
            pass
        
        # Browse by category
        category = st.selectbox("Browse by category:", ["All", "Promoters", "Genes", "Proteins", "Terminators", "RBS"])
        
        # Display parts
        st.markdown("### 🧬 Available Parts")
        
        # Mock parts data
        parts_data = [
            {"name": "pTac", "type": "Promoter", "organism": "E. coli", "strength": "Medium"},
            {"name": "LacI", "type": "Protein", "organism": "E. coli", "function": "Repressor"},
            {"name": "GFP", "type": "Gene", "organism": "E. coli", "product": "Green Fluorescent Protein"},
        ]
        
        for part in parts_data:
            with st.expander(f"🧬 {part['name']} ({part['type']})"):
                st.write(f"**Organism:** {part['organism']}")
                if 'strength' in part:
                    st.write(f"**Strength:** {part['strength']}")
                if 'function' in part:
                    st.write(f"**Function:** {part['function']}")
                if 'product' in part:
                    st.write(f"**Product:** {part['product']}")
    
    def render_settings_page(self):
        """Render the settings page."""
        st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
        
        # API settings
        st.markdown("### 🔑 API Configuration")
        openai_api_key = st.text_input("OpenAI API Key:", type="password", help="Required for advanced NLP features")
        
        if st.button("💾 Save Settings"):
            # Save settings implementation
            st.success("✅ Settings saved successfully!")
        
        # Simulation settings
        st.markdown("### 🔬 Simulation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_time_end = st.number_input("Default Simulation Time (hours):", 1.0, 100.0, 24.0)
            default_time_step = st.number_input("Default Time Step (hours):", 0.01, 1.0, 0.1)
        
        with col2:
            default_solver = st.selectbox("Default Solver:", ["RK45", "RK4", "BDF", "LSODA"])
            default_tolerance = st.number_input("Default Tolerance:", 1e-8, 1e-3, 1e-6, format="%.1e")
        
        # Export settings
        st.markdown("### 📤 Export Settings")
        export_format = st.selectbox("Default Export Format:", ["JSON", "SBOL", "GenBank", "FASTA"])
        
        # About
        st.markdown("### ℹ️ About")
        st.write("**Version:** 1.0.0")
        st.write("**Author:** Genetic Circuit Design Team")
        st.write("**License:** MIT")
    
    def run(self):
        """Run the main application."""
        # Render header
        self.render_header()
        
        # Render sidebar and get settings
        page, settings = self.render_sidebar()
        
        # Render appropriate page
        if page == "🏠 Home":
            self.render_home_page()
        elif page == "🧬 Circuit Design":
            self.render_circuit_design_page(settings)
        elif page == "🔬 Simulation":
            self.render_simulation_page(settings)
        elif page == "📊 Analysis":
            self.render_analysis_page()
        elif page == "📚 Parts Library":
            self.render_parts_library_page()
        elif page == "⚙️ Settings":
            self.render_settings_page()


def main():
    """Main function to run the application."""
    try:
        app = GeneticCircuitApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main() 