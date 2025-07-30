"""
Circuit Designer Module for Genetic Circuit Design Platform.

This module provides advanced algorithms for designing genetic circuits
from parsed logic expressions, including optimization, constraint satisfaction,
and multi-objective design strategies.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
from scipy.optimize import minimize, differential_evolution
from sklearn.cluster import KMeans
from models.circuit import Circuit, CircuitNode, CircuitEdge, LogicGate, LogicGateType, NodeType, EdgeType
from models.parts import GeneticPart, Promoter, Gene, Terminator, RibosomeBindingSite, Organism
from models.simulation import SimulationParameters, SimulationResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DesignConstraint:
    """Represents a design constraint for circuit optimization."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraint_type: str = ""  # equality, inequality, bound
    expression: str = ""
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    weight: float = 1.0
    description: str = ""
    
    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate the constraint with given parameters."""
        # Simple constraint evaluation - can be extended with symbolic computation
        try:
            # Replace parameter names with values in expression
            expr = self.expression
            for param, value in parameters.items():
                expr = expr.replace(param, str(value))
            
            # Evaluate expression (simplified - should use safe eval)
            result = eval(expr)
            
            if self.constraint_type == "equality":
                return abs(result)
            elif self.constraint_type == "inequality":
                return max(0, result)
            else:
                return result
                
        except Exception as e:
            logger.warning(f"Constraint evaluation failed: {e}")
            return np.inf


@dataclass
class DesignObjective:
    """Represents a design objective for circuit optimization."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    objective_type: str = ""  # minimize, maximize
    expression: str = ""
    weight: float = 1.0
    target_value: Optional[float] = None
    description: str = ""
    
    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate the objective with given parameters."""
        try:
            # Replace parameter names with values in expression
            expr = self.expression
            for param, value in parameters.items():
                expr = expr.replace(param, str(value))
            
            # Evaluate expression
            result = eval(expr)
            
            if self.objective_type == "maximize":
                return -result  # Convert to minimization
            else:
                return result
                
        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return np.inf


class CircuitDesigner:
    """Advanced circuit designer with optimization capabilities."""
    
    def __init__(self):
        """Initialize the circuit designer."""
        self.design_templates = self._load_design_templates()
        self.optimization_algorithms = self._load_optimization_algorithms()
        self.constraint_solver = self._load_constraint_solver()
        
        # Design parameters
        self.max_components = 50
        self.max_connections = 100
        self.complexity_threshold = 0.8
        
        logger.info("Circuit designer initialized successfully")
    
    def _load_design_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined circuit design templates."""
        return {
            "simple_repression": {
                "components": ["promoter", "repressor", "gene", "terminator"],
                "connections": [
                    ("promoter", "gene", "transcription"),
                    ("repressor", "promoter", "repression")
                ],
                "parameters": {
                    "promoter_strength": 1.0,
                    "repression_strength": 0.8,
                    "degradation_rate": 0.1
                }
            },
            "simple_activation": {
                "components": ["promoter", "activator", "gene", "terminator"],
                "connections": [
                    ("promoter", "gene", "transcription"),
                    ("activator", "promoter", "activation")
                ],
                "parameters": {
                    "promoter_strength": 1.0,
                    "activation_strength": 1.2,
                    "degradation_rate": 0.1
                }
            },
            "and_gate": {
                "components": ["promoter1", "promoter2", "repressor1", "repressor2", "gene", "terminator"],
                "connections": [
                    ("promoter1", "gene", "transcription"),
                    ("promoter2", "gene", "transcription"),
                    ("repressor1", "promoter1", "repression"),
                    ("repressor2", "promoter2", "repression")
                ],
                "parameters": {
                    "promoter1_strength": 1.0,
                    "promoter2_strength": 1.0,
                    "repression1_strength": 0.8,
                    "repression2_strength": 0.8,
                    "degradation_rate": 0.1
                }
            },
            "or_gate": {
                "components": ["promoter1", "promoter2", "gene1", "gene2", "terminator"],
                "connections": [
                    ("promoter1", "gene1", "transcription"),
                    ("promoter2", "gene2", "transcription")
                ],
                "parameters": {
                    "promoter1_strength": 1.0,
                    "promoter2_strength": 1.0,
                    "degradation_rate": 0.1
                }
            },
            "oscillator": {
                "components": ["promoter1", "promoter2", "repressor1", "repressor2", "gene1", "gene2", "terminator"],
                "connections": [
                    ("promoter1", "gene1", "transcription"),
                    ("promoter2", "gene2", "transcription"),
                    ("repressor1", "promoter2", "repression"),
                    ("repressor2", "promoter1", "repression")
                ],
                "parameters": {
                    "promoter1_strength": 1.0,
                    "promoter2_strength": 1.0,
                    "repression1_strength": 0.9,
                    "repression2_strength": 0.9,
                    "degradation_rate": 0.2,
                    "delay": 1.0
                }
            }
        }
    
    def _load_optimization_algorithms(self) -> Dict[str, Any]:
        """Load optimization algorithms."""
        return {
            "genetic_algorithm": self._genetic_algorithm_optimizer,
            "particle_swarm": self._particle_swarm_optimizer,
            "simulated_annealing": self._simulated_annealing_optimizer,
            "gradient_descent": self._gradient_descent_optimizer,
            "bayesian_optimization": self._bayesian_optimizer
        }
    
    def _load_constraint_solver(self) -> Any:
        """Load constraint solver."""
        # Simplified constraint solver - can be extended with more sophisticated solvers
        return self._simple_constraint_solver
    
    def design_circuit(self, parsing_results: Dict[str, Any]) -> Circuit:
        """Design a genetic circuit from parsing results."""
        logger.info("Starting circuit design process")
        
        # Extract information from parsing results
        parsed_logic = parsing_results.get("parsed_logic", {})
        template = parsing_results.get("template", "simple_repression")
        organism = parsing_results.get("metadata", {}).get("organism", "E. coli")
        
        # Create initial circuit structure
        circuit = self._create_initial_circuit(parsed_logic, template, organism)
        
        # Optimize circuit design
        optimized_circuit = self._optimize_circuit(circuit, parsed_logic)
        
        # Validate design
        validation_errors = self._validate_design(optimized_circuit)
        if validation_errors:
            logger.warning(f"Design validation warnings: {validation_errors}")
        
        logger.info("Circuit design completed successfully")
        return optimized_circuit
    
    def _create_initial_circuit(self, parsed_logic: Dict[str, Any], 
                              template: str, organism: str) -> Circuit:
        """Create initial circuit structure from template."""
        template_info = self.design_templates.get(template, self.design_templates["simple_repression"])
        
        # Create circuit
        circuit = Circuit(
            name=f"Circuit_{template}_{uuid.uuid4().hex[:8]}",
            organism=organism,
            description=f"Generated from {template} template"
        )
        
        # Add components
        component_map = {}
        for i, component_type in enumerate(template_info["components"]):
            node = self._create_component_node(component_type, i, organism)
            circuit.topology.add_node(node)
            component_map[component_type] = node.id
        
        # Add connections
        for source_type, target_type, connection_type in template_info["connections"]:
            if source_type in component_map and target_type in component_map:
                edge = CircuitEdge(
                    source_id=component_map[source_type],
                    target_id=component_map[target_type],
                    edge_type=self._get_edge_type(connection_type)
                )
                circuit.topology.add_edge(edge)
        
        # Add parameters
        circuit.parameters.update(template_info["parameters"])
        
        return circuit
    
    def _create_component_node(self, component_type: str, index: int, organism: str) -> CircuitNode:
        """Create a component node."""
        node_types = {
            "promoter": NodeType.PROMOTER,
            "gene": NodeType.GENE,
            "terminator": NodeType.TERMINATOR,
            "repressor": NodeType.REGULATOR,
            "activator": NodeType.REGULATOR,
            "rbs": NodeType.RBS
        }
        
        node_type = node_types.get(component_type, NodeType.GENE)
        
        return CircuitNode(
            name=f"{component_type}_{index}",
            node_type=node_type,
            position=(index * 100, 0),
            properties={
                "organism": organism,
                "component_type": component_type,
                "strength": 1.0
            }
        )
    
    def _get_edge_type(self, connection_type: str) -> EdgeType:
        """Get edge type from connection type."""
        edge_type_map = {
            "activation": EdgeType.ACTIVATION,
            "repression": EdgeType.INHIBITION,  # Using INHIBITION instead of REPRESSION
            "transcription": EdgeType.TRANSCRIPTION,
            "translation": EdgeType.TRANSLATION,
            "degradation": EdgeType.DEGRADATION,
            "binding": EdgeType.BINDING,
            "catalysis": EdgeType.CATALYSIS,
            "regulation": EdgeType.REGULATION
        }
        return edge_type_map.get(connection_type.lower(), EdgeType.REGULATION)
    
    def _optimize_circuit(self, circuit: Circuit, parsed_logic: Dict[str, Any]) -> Circuit:
        """Optimize circuit design using multiple objectives."""
        logger.info("Starting circuit optimization")
        
        # Define objectives
        objectives = self._define_objectives(circuit, parsed_logic)
        
        # Define constraints
        constraints = self._define_constraints(circuit)
        
        # Run multi-objective optimization
        optimized_parameters = self._multi_objective_optimization(
            circuit, objectives, constraints
        )
        
        # Apply optimized parameters
        circuit.parameters.update(optimized_parameters)
        
        # Optimize topology if needed
        if len(circuit.topology.nodes) > 10:
            circuit = self._optimize_topology(circuit)
        
        logger.info("Circuit optimization completed")
        return circuit
    
    def _define_objectives(self, circuit: Circuit, parsed_logic: Dict[str, Any]) -> List[DesignObjective]:
        """Define optimization objectives."""
        objectives = []
        
        # Performance objective
        performance_obj = DesignObjective(
            name="performance",
            objective_type="maximize",
            expression="output_expression",
            weight=1.0,
            description="Maximize output expression"
        )
        objectives.append(performance_obj)
        
        # Robustness objective
        robustness_obj = DesignObjective(
            name="robustness",
            objective_type="maximize",
            expression="1 / (1 + parameter_sensitivity)",
            weight=0.8,
            description="Maximize robustness to parameter variations"
        )
        objectives.append(robustness_obj)
        
        # Complexity objective (minimize)
        complexity_obj = DesignObjective(
            name="complexity",
            objective_type="minimize",
            expression="num_components + num_connections",
            weight=0.5,
            description="Minimize circuit complexity"
        )
        objectives.append(complexity_obj)
        
        # Resource usage objective
        resource_obj = DesignObjective(
            name="resource_usage",
            objective_type="minimize",
            expression="total_expression_level",
            weight=0.3,
            description="Minimize resource usage"
        )
        objectives.append(resource_obj)
        
        return objectives
    
    def _define_constraints(self, circuit: Circuit) -> List[DesignConstraint]:
        """Define design constraints."""
        constraints = []
        
        # Expression level constraints
        expression_constraint = DesignConstraint(
            constraint_type="inequality",
            expression="output_expression - 0.1",
            lower_bound=0,
            description="Minimum output expression"
        )
        constraints.append(expression_constraint)
        
        # Stability constraints
        stability_constraint = DesignConstraint(
            constraint_type="inequality",
            expression="1 - parameter_sensitivity",
            lower_bound=0,
            description="System stability"
        )
        constraints.append(stability_constraint)
        
        # Component limit constraints
        component_constraint = DesignConstraint(
            constraint_type="inequality",
            expression=f"{self.max_components} - num_components",
            lower_bound=0,
            description="Component limit"
        )
        constraints.append(component_constraint)
        
        return constraints
    
    def _multi_objective_optimization(self, circuit: Circuit, 
                                    objectives: List[DesignObjective],
                                    constraints: List[DesignConstraint]) -> Dict[str, float]:
        """Perform multi-objective optimization."""
        logger.info("Running multi-objective optimization")
        
        # Get current parameters
        current_params = circuit.parameters.copy()
        param_names = list(current_params.keys())
        param_values = list(current_params.values())
        
        # Define objective function
        def objective_function(x):
            # Update parameters
            params = dict(zip(param_names, x))
            
            # Evaluate objectives
            objective_values = []
            for obj in objectives:
                value = obj.evaluate(params)
                objective_values.append(value * obj.weight)
            
            # Evaluate constraints
            constraint_penalty = 0
            for constraint in constraints:
                penalty = constraint.evaluate(params)
                constraint_penalty += penalty * constraint.weight
            
            # Return weighted sum (can be extended to Pareto optimization)
            return sum(objective_values) + constraint_penalty
        
        # Run optimization
        try:
            result = minimize(
                objective_function,
                param_values,
                method='L-BFGS-B',
                bounds=[(0.1, 10.0)] * len(param_values),
                options={'maxiter': 100}
            )
            
            if result.success:
                optimized_params = dict(zip(param_names, result.x))
                logger.info(f"Optimization successful: {result.message}")
                return optimized_params
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return current_params
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return current_params
    
    def _optimize_topology(self, circuit: Circuit) -> Circuit:
        """Optimize circuit topology for better performance."""
        logger.info("Optimizing circuit topology")
        
        # Create graph representation
        G = circuit.topology.to_networkx()
        
        # Analyze topology
        if nx.is_weakly_connected(G):
            # Optimize layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Update node positions
            for node in circuit.topology.nodes:
                if node.id in pos:
                    node.position = (float(pos[node.id][0]), float(pos[node.id][1]))
        
        # Remove redundant connections
        circuit = self._remove_redundant_connections(circuit)
        
        # Optimize component placement
        circuit = self._optimize_component_placement(circuit)
        
        return circuit
    
    def _remove_redundant_connections(self, circuit: Circuit) -> Circuit:
        """Remove redundant connections from circuit."""
        G = circuit.topology.to_networkx()
        
        # Find redundant edges (multiple paths between same nodes)
        redundant_edges = []
        
        for edge in circuit.topology.edges:
            # Check if there's another path between source and target
            G_temp = G.copy()
            G_temp.remove_edge(edge.source_id, edge.target_id)
            
            if nx.has_path(G_temp, edge.source_id, edge.target_id):
                redundant_edges.append(edge.id)
        
        # Remove redundant edges
        for edge_id in redundant_edges:
            circuit.topology.remove_edge(edge_id)
        
        return circuit
    
    def _optimize_component_placement(self, circuit: Circuit) -> Circuit:
        """Optimize component placement for better visualization."""
        # Use clustering to group related components
        if len(circuit.topology.nodes) > 3:
            positions = np.array([node.position for node in circuit.topology.nodes])
            
            # Cluster components
            kmeans = KMeans(n_clusters=min(5, len(circuit.topology.nodes)), random_state=42)
            clusters = kmeans.fit_predict(positions)
            
            # Update positions based on clusters
            for i, node in enumerate(circuit.topology.nodes):
                cluster_center = kmeans.cluster_centers_[clusters[i]]
                node.position = (float(cluster_center[0]), float(cluster_center[1]))
        
        return circuit
    
    def _validate_design(self, circuit: Circuit) -> List[str]:
        """Validate circuit design and return error messages."""
        errors = []
        
        # Check component limits
        if len(circuit.topology.nodes) > self.max_components:
            errors.append(f"Too many components: {len(circuit.topology.nodes)} > {self.max_components}")
        
        if len(circuit.topology.edges) > self.max_connections:
            errors.append(f"Too many connections: {len(circuit.topology.edges)} > {self.max_connections}")
        
        # Check connectivity
        G = circuit.topology.to_networkx()
        if len(G.nodes) > 1 and not nx.is_weakly_connected(G):
            errors.append("Circuit is not connected")
        
        # Check for cycles (if not desired)
        cycles = list(nx.simple_cycles(G))
        if cycles and not self._is_oscillator_design(circuit):
            errors.append(f"Unexpected cycles detected: {cycles}")
        
        # Check parameter bounds
        for param, value in circuit.parameters.items():
            if value < 0:
                errors.append(f"Negative parameter value: {param} = {value}")
            if value > 100:
                errors.append(f"Parameter value too high: {param} = {value}")
        
        return errors
    
    def _is_oscillator_design(self, circuit: Circuit) -> bool:
        """Check if circuit is designed for oscillatory behavior."""
        # Check if circuit has feedback loops
        G = circuit.topology.to_networkx()
        cycles = list(nx.simple_cycles(G))
        
        # Check for oscillatory parameters
        oscillatory_params = ["delay", "oscillation_period", "feedback_strength"]
        has_oscillatory_params = any(param in circuit.parameters for param in oscillatory_params)
        
        return len(cycles) > 0 or has_oscillatory_params
    
    def _genetic_algorithm_optimizer(self, objective_func, bounds, **kwargs):
        """Genetic algorithm optimizer."""
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=kwargs.get('maxiter', 100),
            popsize=kwargs.get('popsize', 15),
            seed=42
        )
        return result
    
    def _particle_swarm_optimizer(self, objective_func, bounds, **kwargs):
        """Particle swarm optimizer."""
        # Simplified implementation - can be extended with proper PSO
        return self._genetic_algorithm_optimizer(objective_func, bounds, **kwargs)
    
    def _simulated_annealing_optimizer(self, objective_func, bounds, **kwargs):
        """Simulated annealing optimizer."""
        # Simplified implementation - can be extended with proper SA
        return self._genetic_algorithm_optimizer(objective_func, bounds, **kwargs)
    
    def _gradient_descent_optimizer(self, objective_func, bounds, **kwargs):
        """Gradient descent optimizer."""
        # Simplified implementation - can be extended with proper GD
        return self._genetic_algorithm_optimizer(objective_func, bounds, **kwargs)
    
    def _bayesian_optimizer(self, objective_func, bounds, **kwargs):
        """Bayesian optimization."""
        # Simplified implementation - can be extended with proper BO
        return self._genetic_algorithm_optimizer(objective_func, bounds, **kwargs)
    
    def _simple_constraint_solver(self, constraints, **kwargs):
        """Simple constraint solver."""
        # Simplified implementation - can be extended with proper constraint solvers
        return True


class CircuitOptimizer:
    """Specialized circuit optimizer for advanced optimization strategies."""
    
    def __init__(self):
        """Initialize the circuit optimizer."""
        self.optimization_strategies = {
            "performance": self._optimize_performance,
            "robustness": self._optimize_robustness,
            "efficiency": self._optimize_efficiency,
            "stability": self._optimize_stability
        }
    
    def optimize_circuit(self, circuit: Circuit, strategy: str = "performance") -> Circuit:
        """Optimize circuit using specified strategy."""
        if strategy in self.optimization_strategies:
            return self.optimization_strategies[strategy](circuit)
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
            return circuit
    
    def _optimize_performance(self, circuit: Circuit) -> Circuit:
        """Optimize circuit for maximum performance."""
        # Implementation for performance optimization
        return circuit
    
    def _optimize_robustness(self, circuit: Circuit) -> Circuit:
        """Optimize circuit for maximum robustness."""
        # Implementation for robustness optimization
        return circuit
    
    def _optimize_efficiency(self, circuit: Circuit) -> Circuit:
        """Optimize circuit for maximum efficiency."""
        # Implementation for efficiency optimization
        return circuit
    
    def _optimize_stability(self, circuit: Circuit) -> Circuit:
        """Optimize circuit for maximum stability."""
        # Implementation for stability optimization
        return circuit 