"""
Validation utilities for genetic circuit design.

This module provides validation functions for circuits, parameters,
sequences, and other biological components.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from models.circuit import Circuit, CircuitNode, CircuitEdge, NodeType, EdgeType
from models.parts import GeneticPart, PartType, Organism
from models.simulation import SimulationParameters

logger = logging.getLogger(__name__)


def validate_circuit(circuit: Circuit) -> Dict[str, Any]:
    """
    Comprehensive circuit validation.
    
    Args:
        circuit: Circuit to validate
        
    Returns:
        Validation results with errors and warnings
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Basic circuit validation
    basic_errors = _validate_basic_circuit(circuit)
    results["errors"].extend(basic_errors)
    
    # Topology validation
    topology_errors = _validate_circuit_topology(circuit)
    results["errors"].extend(topology_errors)
    
    # Component validation
    component_errors = _validate_circuit_components(circuit)
    results["errors"].extend(component_errors)
    
    # Parameter validation
    parameter_errors = _validate_circuit_parameters(circuit)
    results["errors"].extend(parameter_errors)
    
    # Biological validation
    biological_errors = _validate_biological_feasibility(circuit)
    results["warnings"].extend(biological_errors)
    
    # Performance validation
    performance_suggestions = _validate_circuit_performance(circuit)
    results["suggestions"].extend(performance_suggestions)
    
    # Update validity
    results["valid"] = len(results["errors"]) == 0
    
    return results


def _validate_basic_circuit(circuit: Circuit) -> List[str]:
    """Validate basic circuit properties."""
    errors = []
    
    # Check required fields
    if not circuit.name:
        errors.append("Circuit must have a name")
    
    if not circuit.organism:
        errors.append("Circuit must specify target organism")
    
    # Check component limits
    if len(circuit.topology.nodes) == 0:
        errors.append("Circuit must have at least one component")
    
    if len(circuit.topology.nodes) > 100:
        errors.append("Circuit has too many components (>100)")
    
    if len(circuit.topology.edges) > 200:
        errors.append("Circuit has too many connections (>200)")
    
    return errors


def _validate_circuit_topology(circuit: Circuit) -> List[str]:
    """Validate circuit topology."""
    errors = []
    
    G = circuit.topology.to_networkx()
    
    # Check for disconnected components
    if len(G.nodes) > 1:
        if not nx.is_weakly_connected(G):
            errors.append("Circuit contains disconnected components")
    
    # Check for self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        errors.append(f"Circuit contains {len(self_loops)} self-loops")
    
    # Check for cycles (if not intended)
    cycles = list(nx.simple_cycles(G))
    if cycles and not _is_oscillator_design(circuit):
        errors.append(f"Circuit contains {len(cycles)} cycles (may cause instability)")
    
    # Check for orphaned nodes
    orphaned_nodes = []
    for node in circuit.topology.nodes:
        connected_edges = circuit.topology.get_edges_by_node(node.id)
        if not connected_edges:
            orphaned_nodes.append(node.name)
    
    if orphaned_nodes:
        errors.append(f"Orphaned nodes found: {orphaned_nodes}")
    
    return errors


def _validate_circuit_components(circuit: Circuit) -> List[str]:
    """Validate circuit components."""
    errors = []
    
    # Check component types
    valid_node_types = [node_type.value for node_type in NodeType]
    valid_edge_types = [edge_type.value for edge_type in EdgeType]
    
    for node in circuit.topology.nodes:
        if node.node_type.value not in valid_node_types:
            errors.append(f"Invalid node type '{node.node_type.value}' for node '{node.name}'")
        
        # Check for duplicate names
        duplicate_names = [n for n in circuit.topology.nodes if n.name == node.name and n.id != node.id]
        if duplicate_names:
            errors.append(f"Duplicate node name '{node.name}'")
    
    for edge in circuit.topology.edges:
        if edge.edge_type.value not in valid_edge_types:
            errors.append(f"Invalid edge type '{edge.edge_type.value}' for edge {edge.id}")
        
        # Check edge connections
        source_node = circuit.topology.get_node_by_id(edge.source_id)
        target_node = circuit.topology.get_node_by_id(edge.target_id)
        
        if not source_node:
            errors.append(f"Edge {edge.id} references non-existent source node {edge.source_id}")
        
        if not target_node:
            errors.append(f"Edge {edge.id} references non-existent target node {edge.target_id}")
    
    return errors


def _validate_circuit_parameters(circuit: Circuit) -> List[str]:
    """Validate circuit parameters."""
    errors = []
    
    for param_name, param_value in circuit.parameters.items():
        # Check parameter types
        if not isinstance(param_value, (int, float)):
            errors.append(f"Parameter '{param_name}' must be numeric, got {type(param_value)}")
            continue
        
        # Check parameter bounds
        if param_value < 0:
            errors.append(f"Parameter '{param_name}' cannot be negative: {param_value}")
        
        if param_value > 1000:
            errors.append(f"Parameter '{param_name}' seems too large: {param_value}")
        
        # Check for reasonable ranges based on parameter type
        if "rate" in param_name.lower():
            if param_value > 100:
                errors.append(f"Rate parameter '{param_name}' seems too high: {param_value}")
        
        if "concentration" in param_name.lower():
            if param_value > 1000:
                errors.append(f"Concentration parameter '{param_name}' seems too high: {param_value}")
    
    return errors


def _validate_biological_feasibility(circuit: Circuit) -> List[str]:
    """Validate biological feasibility."""
    warnings = []
    
    # Check for unrealistic parameter combinations
    if "transcription_rate" in circuit.parameters and "translation_rate" in circuit.parameters:
        tr_rate = circuit.parameters["transcription_rate"]
        tl_rate = circuit.parameters["translation_rate"]
        
        if tl_rate > tr_rate * 10:
            warnings.append("Translation rate is much higher than transcription rate")
    
    # Check for resource competition
    protein_genes = [node for node in circuit.topology.nodes if node.node_type == NodeType.GENE]
    if len(protein_genes) > 20:
        warnings.append("Many protein-coding genes may cause resource competition")
    
    # Check for regulatory complexity
    regulatory_edges = [edge for edge in circuit.topology.edges 
                       if edge.edge_type in [EdgeType.REPRESSION, EdgeType.ACTIVATION]]
    if len(regulatory_edges) > 50:
        warnings.append("High regulatory complexity may cause instability")
    
    return warnings


def _validate_circuit_performance(circuit: Circuit) -> List[str]:
    """Validate circuit performance characteristics."""
    suggestions = []
    
    # Check for potential bottlenecks
    promoter_nodes = [node for node in circuit.topology.nodes if node.node_type == NodeType.PROMOTER]
    if len(promoter_nodes) == 0:
        suggestions.append("Consider adding promoters for transcriptional control")
    
    # Check for termination
    terminator_nodes = [node for node in circuit.topology.nodes if node.node_type == NodeType.TERMINATOR]
    if len(terminator_nodes) == 0:
        suggestions.append("Consider adding terminators for proper transcription termination")
    
    # Check for RBS
    rbs_nodes = [node for node in circuit.topology.nodes if node.node_type == NodeType.RBS]
    if len(rbs_nodes) == 0:
        suggestions.append("Consider adding ribosome binding sites for translation control")
    
    # Check for reporters
    reporter_nodes = [node for node in circuit.topology.nodes if "reporter" in node.name.lower()]
    if len(reporter_nodes) == 0:
        suggestions.append("Consider adding reporter genes for monitoring circuit performance")
    
    return suggestions


def _is_oscillator_design(circuit: Circuit) -> bool:
    """Check if circuit is designed for oscillatory behavior."""
    # Check for feedback loops
    G = circuit.topology.to_networkx()
    cycles = list(nx.simple_cycles(G))
    
    # Check for oscillatory parameters
    oscillatory_params = ["delay", "oscillation_period", "feedback_strength"]
    has_oscillatory_params = any(param in circuit.parameters for param in oscillatory_params)
    
    # Check for oscillatory keywords in description
    oscillatory_keywords = ["oscillat", "rhythm", "cycle", "periodic"]
    has_oscillatory_keywords = any(keyword in circuit.description.lower() 
                                 for keyword in oscillatory_keywords)
    
    return len(cycles) > 0 or has_oscillatory_params or has_oscillatory_keywords


def validate_parameters(parameters: Dict[str, float], parameter_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parameter values against specifications.
    
    Args:
        parameters: Parameter values to validate
        parameter_specs: Parameter specifications with bounds and types
        
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    for param_name, param_value in parameters.items():
        if param_name not in parameter_specs:
            results["warnings"].append(f"Unknown parameter: {param_name}")
            continue
        
        spec = parameter_specs[param_name]
        
        # Check type
        expected_type = spec.get("type", float)
        if not isinstance(param_value, expected_type):
            results["errors"].append(f"Parameter '{param_name}' should be {expected_type}, got {type(param_value)}")
            results["valid"] = False
            continue
        
        # Check bounds
        if "min" in spec and param_value < spec["min"]:
            results["errors"].append(f"Parameter '{param_name}' below minimum: {param_value} < {spec['min']}")
            results["valid"] = False
        
        if "max" in spec and param_value > spec["max"]:
            results["errors"].append(f"Parameter '{param_name}' above maximum: {param_value} > {spec['max']}")
            results["valid"] = False
        
        # Check allowed values
        if "allowed_values" in spec and param_value not in spec["allowed_values"]:
            results["errors"].append(f"Parameter '{param_name}' value {param_value} not in allowed values: {spec['allowed_values']}")
            results["valid"] = False
    
    return results


def validate_sequence(sequence: str, sequence_type: str = "DNA") -> Dict[str, Any]:
    """
    Validate biological sequence.
    
    Args:
        sequence: Sequence to validate
        sequence_type: Type of sequence ("DNA", "RNA", "PROTEIN")
        
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    if not sequence:
        results["errors"].append("Sequence cannot be empty")
        results["valid"] = False
        return results
    
    # Check length
    if len(sequence) < 10:
        results["warnings"].append("Sequence is very short")
    
    if len(sequence) > 10000:
        results["warnings"].append("Sequence is very long")
    
    # Check characters based on type
    sequence_upper = sequence.upper()
    
    if sequence_type == "DNA":
        valid_chars = set("ATCGN")
        invalid_chars = set(sequence_upper) - valid_chars
        if invalid_chars:
            results["errors"].append(f"Invalid DNA characters: {invalid_chars}")
            results["valid"] = False
        
        # Check for ambiguous bases
        if "N" in sequence_upper:
            results["warnings"].append("Sequence contains ambiguous bases (N)")
    
    elif sequence_type == "RNA":
        valid_chars = set("AUCGN")
        invalid_chars = set(sequence_upper) - valid_chars
        if invalid_chars:
            results["errors"].append(f"Invalid RNA characters: {invalid_chars}")
            results["valid"] = False
    
    elif sequence_type == "PROTEIN":
        valid_chars = set("ACDEFGHIKLMNPQRSTVWY*")
        invalid_chars = set(sequence_upper) - valid_chars
        if invalid_chars:
            results["errors"].append(f"Invalid protein characters: {invalid_chars}")
            results["valid"] = False
        
        # Check for stop codons
        if "*" in sequence_upper:
            results["warnings"].append("Sequence contains stop codons (*)")
    
    # Check for repeated patterns
    if len(sequence) > 20:
        for pattern_length in [3, 6, 9]:
            patterns = {}
            for i in range(len(sequence) - pattern_length + 1):
                pattern = sequence[i:i + pattern_length]
                patterns[pattern] = patterns.get(pattern, 0) + 1
            
            repeated_patterns = [pattern for pattern, count in patterns.items() if count > 3]
            if repeated_patterns:
                results["warnings"].append(f"Repeated patterns of length {pattern_length}: {repeated_patterns[:3]}")
    
    return results


def validate_organism(organism: str) -> Dict[str, Any]:
    """
    Validate organism specification.
    
    Args:
        organism: Organism name to validate
        
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    valid_organisms = [
        "E. coli", "Escherichia coli",
        "S. cerevisiae", "Saccharomyces cerevisiae",
        "B. subtilis", "Bacillus subtilis",
        "P. putida", "Pseudomonas putida",
        "mammalian", "human", "mouse", "rat"
    ]
    
    if organism not in valid_organisms:
        results["warnings"].append(f"Unknown organism: {organism}")
        results["warnings"].append(f"Supported organisms: {valid_organisms}")
    
    return results


def validate_simulation_parameters(params: SimulationParameters) -> Dict[str, Any]:
    """
    Validate simulation parameters.
    
    Args:
        params: Simulation parameters to validate
        
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check time parameters
    if params.time_start >= params.time_end:
        results["errors"].append("Time start must be before time end")
        results["valid"] = False
    
    if params.time_step <= 0:
        results["errors"].append("Time step must be positive")
        results["valid"] = False
    
    if params.time_step > (params.time_end - params.time_start):
        results["errors"].append("Time step is larger than simulation duration")
        results["valid"] = False
    
    # Check tolerance
    if params.tolerance <= 0:
        results["errors"].append("Tolerance must be positive")
        results["valid"] = False
    
    if params.tolerance > 1:
        results["warnings"].append("Tolerance seems very large")
    
    # Check max steps
    if params.max_steps <= 0:
        results["errors"].append("Max steps must be positive")
        results["valid"] = False
    
    if params.max_steps > 100000:
        results["warnings"].append("Max steps is very large")
    
    # Check initial conditions
    for var_name, value in params.initial_conditions.items():
        if not isinstance(value, (int, float)):
            results["errors"].append(f"Initial condition '{var_name}' must be numeric")
            results["valid"] = False
        elif value < 0:
            results["warnings"].append(f"Initial condition '{var_name}' is negative")
    
    return results


def validate_genetic_part(part: GeneticPart) -> Dict[str, Any]:
    """
    Validate genetic part.
    
    Args:
        part: Genetic part to validate
        
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required fields
    if not part.name:
        results["errors"].append("Part must have a name")
        results["valid"] = False
    
    if not part.part_type:
        results["errors"].append("Part must have a type")
        results["valid"] = False
    
    if not part.organism:
        results["errors"].append("Part must have an organism")
        results["valid"] = False
    
    # Validate sequence if present
    if part.sequence and part.sequence.sequence:
        sequence_type = "DNA" if part.part_type in [PartType.PROMOTER, PartType.GENE, PartType.TERMINATOR] else "PROTEIN"
        seq_validation = validate_sequence(part.sequence.sequence, sequence_type)
        
        if not seq_validation["valid"]:
            results["errors"].extend(seq_validation["errors"])
            results["valid"] = False
        
        results["warnings"].extend(seq_validation["warnings"])
    
    # Check properties
    if hasattr(part, 'properties') and part.properties:
        for prop_name, prop_value in part.properties.items():
            if isinstance(prop_value, (int, float)) and prop_value < 0:
                results["warnings"].append(f"Property '{prop_name}' is negative")
    
    return results


def validate_circuit_design_rules(circuit: Circuit) -> Dict[str, Any]:
    """
    Validate circuit against design rules.
    
    Args:
        circuit: Circuit to validate
        
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "rule_violations": []
    }
    
    # Design rule 1: Promoters should have terminators
    promoters = [node for node in circuit.topology.nodes if node.node_type == NodeType.PROMOTER]
    terminators = [node for node in circuit.topology.nodes if node.node_type == NodeType.TERMINATOR]
    
    if len(promoters) > len(terminators):
        results["warnings"].append("More promoters than terminators - consider adding terminators")
    
    # Design rule 2: Genes should have RBS
    genes = [node for node in circuit.topology.nodes if node.node_type == NodeType.GENE]
    rbs_sites = [node for node in circuit.topology.nodes if node.node_type == NodeType.RBS]
    
    if len(genes) > len(rbs_sites):
        results["warnings"].append("More genes than RBS sites - consider adding RBS")
    
    # Design rule 3: Check for proper regulatory connections
    regulatory_edges = [edge for edge in circuit.topology.edges 
                       if edge.edge_type in [EdgeType.REPRESSION, EdgeType.ACTIVATION]]
    
    for edge in regulatory_edges:
        source_node = circuit.topology.get_node_by_id(edge.source_id)
        target_node = circuit.topology.get_node_by_id(edge.target_id)
        
        if source_node and target_node:
            # Repressors should target promoters
            if edge.edge_type == EdgeType.REPRESSION and target_node.node_type != NodeType.PROMOTER:
                results["warnings"].append(f"Repressor {source_node.name} targets non-promoter {target_node.name}")
            
            # Activators should target promoters
            if edge.edge_type == EdgeType.ACTIVATION and target_node.node_type != NodeType.PROMOTER:
                results["warnings"].append(f"Activator {source_node.name} targets non-promoter {target_node.name}")
    
    # Design rule 4: Check for resource competition
    if len(genes) > 10:
        results["warnings"].append("Many genes may cause resource competition")
    
    # Design rule 5: Check for feedback loops
    G = circuit.topology.to_networkx()
    cycles = list(nx.simple_cycles(G))
    
    if len(cycles) > 3:
        results["warnings"].append("Many feedback loops may cause instability")
    
    return results 