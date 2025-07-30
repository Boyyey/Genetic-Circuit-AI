"""
Circuit data models for genetic circuit representation.

This module defines the core data structures for representing genetic circuits,
including nodes, edges, topologies, and logic gates. It provides a flexible
framework for modeling complex biological systems with regulatory interactions.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import networkx as nx
from pydantic import BaseModel, Field, validator
import numpy as np


class NodeType(Enum):
    """Types of nodes in a genetic circuit."""
    PROMOTER = "promoter"
    GENE = "gene"
    TERMINATOR = "terminator"
    RBS = "ribosome_binding_site"
    PROTEIN = "protein"
    SMALL_MOLECULE = "small_molecule"
    LOGIC_GATE = "logic_gate"
    SENSOR = "sensor"
    REPORTER = "reporter"
    REGULATOR = "regulator"


class EdgeType(Enum):
    """Types of edges in a genetic circuit."""
    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    REGULATION = "regulation"
    DEGRADATION = "degradation"
    BINDING = "binding"
    CATALYSIS = "catalysis"
    INHIBITION = "inhibition"
    ACTIVATION = "activation"


class LogicGateType(Enum):
    """Types of logic gates."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    NAND = "NAND"
    NOR = "NOR"
    XOR = "XOR"
    XNOR = "XNOR"
    BUFFER = "BUFFER"


class RegulationType(Enum):
    """Types of regulatory interactions."""
    ACTIVATION = "activation"
    REPRESSION = "repression"
    DUAL = "dual"
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"


@dataclass
class CircuitNode:
    """Represents a node in a genetic circuit."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: NodeType = NodeType.GENE
    position: Tuple[float, float] = field(default=(0.0, 0.0))
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and set default properties."""
        if not self.name:
            self.name = f"{self.node_type.value}_{self.id[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "position": self.position,
            "properties": self.properties,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitNode":
        """Create node from dictionary representation."""
        data["node_type"] = NodeType(data["node_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class CircuitEdge:
    """Represents an edge in a genetic circuit."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.REGULATION
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitEdge":
        """Create edge from dictionary representation."""
        data["edge_type"] = EdgeType(data["edge_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class LogicGate:
    """Represents a logic gate in a genetic circuit."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    gate_type: LogicGateType = LogicGateType.AND
    inputs: List[str] = field(default_factory=list)
    output: str = ""
    threshold: float = 0.5
    cooperativity: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, input_values: Dict[str, float]) -> float:
        """Evaluate the logic gate with given input values."""
        if not self.inputs:
            return 0.0
        
        input_signals = [input_values.get(input_id, 0.0) for input_id in self.inputs]
        
        if self.gate_type == LogicGateType.AND:
            return min(input_signals) if all(s >= self.threshold for s in input_signals) else 0.0
        elif self.gate_type == LogicGateType.OR:
            return max(input_signals) if any(s >= self.threshold for s in input_signals) else 0.0
        elif self.gate_type == LogicGateType.NOT:
            return 1.0 - input_signals[0] if len(input_signals) == 1 else 0.0
        elif self.gate_type == LogicGateType.NAND:
            return 1.0 - min(input_signals) if all(s >= self.threshold for s in input_signals) else 1.0
        elif self.gate_type == LogicGateType.NOR:
            return 1.0 - max(input_signals) if any(s >= self.threshold for s in input_signals) else 1.0
        elif self.gate_type == LogicGateType.XOR:
            active_inputs = sum(1 for s in input_signals if s >= self.threshold)
            return max(input_signals) if active_inputs == 1 else 0.0
        elif self.gate_type == LogicGateType.XNOR:
            active_inputs = sum(1 for s in input_signals if s >= self.threshold)
            return max(input_signals) if active_inputs % 2 == 0 else 0.0
        elif self.gate_type == LogicGateType.BUFFER:
            return input_signals[0] if len(input_signals) == 1 else 0.0
        
        return 0.0


@dataclass
class RegulatoryInteraction:
    """Represents a regulatory interaction between components."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    regulator_id: str = ""
    target_id: str = ""
    regulation_type: RegulationType = RegulationType.ACTIVATION
    strength: float = 1.0
    cooperativity: float = 1.0
    dissociation_constant: float = 1.0
    hill_coefficient: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_regulation(self, regulator_concentration: float) -> float:
        """Calculate regulation strength based on regulator concentration."""
        if self.regulation_type == RegulationType.ACTIVATION:
            return (regulator_concentration ** self.hill_coefficient) / (
                self.dissociation_constant ** self.hill_coefficient + 
                regulator_concentration ** self.hill_coefficient
            )
        elif self.regulation_type == RegulationType.REPRESSION:
            return 1.0 - (regulator_concentration ** self.hill_coefficient) / (
                self.dissociation_constant ** self.hill_coefficient + 
                regulator_concentration ** self.hill_coefficient
            )
        elif self.regulation_type == RegulationType.DUAL:
            # Biphasic regulation
            if regulator_concentration < self.dissociation_constant:
                return regulator_concentration / self.dissociation_constant
            else:
                return 1.0 - (regulator_concentration - self.dissociation_constant) / self.dissociation_constant
        else:
            return 1.0


class CircuitTopology(BaseModel):
    """Represents the topology of a genetic circuit."""
    
    nodes: List[CircuitNode] = Field(default_factory=list)
    edges: List[CircuitEdge] = Field(default_factory=list)
    logic_gates: List[LogicGate] = Field(default_factory=list)
    regulatory_interactions: List[RegulatoryInteraction] = Field(default_factory=list)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert topology to NetworkX directed graph."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(node.id, **node.to_dict())
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
        
        return G
    
    def from_networkx(self, G: nx.DiGraph) -> None:
        """Load topology from NetworkX directed graph."""
        self.nodes.clear()
        self.edges.clear()
        
        # Load nodes
        for node_id, node_data in G.nodes(data=True):
            node = CircuitNode.from_dict(node_data)
            self.nodes.append(node)
        
        # Load edges
        for source, target, edge_data in G.edges(data=True):
            edge = CircuitEdge.from_dict(edge_data)
            self.edges.append(edge)
    
    def get_node_by_id(self, node_id: str) -> Optional[CircuitNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_by_node(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges connected to a node."""
        edges = []
        for edge in self.edges:
            if edge.source_id == node_id or edge.target_id == node_id:
                edges.append(edge)
        return edges
    
    def add_node(self, node: CircuitNode) -> None:
        """Add a node to the topology."""
        self.nodes.append(node)
    
    def add_edge(self, edge: CircuitEdge) -> None:
        """Add an edge to the topology."""
        self.edges.append(edge)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node and its connected edges."""
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.source_id != node_id and e.target_id != node_id]
    
    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge from the topology."""
        self.edges = [e for e in self.edges if e.id != edge_id]
    
    def validate(self) -> List[str]:
        """Validate the circuit topology and return error messages."""
        errors = []
        
        # Check for orphaned nodes
        node_ids = {node.id for node in self.nodes}
        edge_node_ids = set()
        for edge in self.edges:
            edge_node_ids.add(edge.source_id)
            edge_node_ids.add(edge.target_id)
        
        orphaned_nodes = node_ids - edge_node_ids
        if orphaned_nodes:
            errors.append(f"Orphaned nodes found: {orphaned_nodes}")
        
        # Check for invalid edges
        for edge in self.edges:
            if edge.source_id not in node_ids:
                errors.append(f"Edge {edge.id} references non-existent source node {edge.source_id}")
            if edge.target_id not in node_ids:
                errors.append(f"Edge {edge.id} references non-existent target node {edge.target_id}")
        
        # Check for cycles (optional validation)
        try:
            G = self.to_networkx()
            cycles = list(nx.simple_cycles(G))
            if cycles:
                errors.append(f"Circuit contains cycles: {cycles}")
        except Exception as e:
            errors.append(f"Error checking for cycles: {e}")
        
        return errors


class Circuit(BaseModel):
    """Main circuit class representing a complete genetic circuit."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="Unnamed Circuit")
    description: str = Field(default="")
    organism: str = Field(default="E. coli")
    topology: CircuitTopology = Field(default_factory=CircuitTopology)
    parameters: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            CircuitNode: lambda v: v.to_dict(),
            CircuitEdge: lambda v: v.to_dict(),
            LogicGate: lambda v: v.__dict__,
            RegulatoryInteraction: lambda v: v.__dict__
        }
    
    def __init__(self, **data):
        """Initialize circuit with validation."""
        super().__init__(**data)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "organism": self.organism,
            "topology": {
                "nodes": [node.to_dict() for node in self.topology.nodes],
                "edges": [edge.to_dict() for edge in self.topology.edges],
                "logic_gates": [gate.__dict__ for gate in self.topology.logic_gates],
                "regulatory_interactions": [reg.__dict__ for reg in self.topology.regulatory_interactions]
            },
            "parameters": self.parameters,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Circuit":
        """Create circuit from dictionary representation."""
        # Reconstruct topology
        topology_data = data.get("topology", {})
        topology = CircuitTopology()
        
        # Reconstruct nodes
        for node_data in topology_data.get("nodes", []):
            node = CircuitNode.from_dict(node_data)
            topology.add_node(node)
        
        # Reconstruct edges
        for edge_data in topology_data.get("edges", []):
            edge = CircuitEdge.from_dict(edge_data)
            topology.add_edge(edge)
        
        # Reconstruct logic gates
        for gate_data in topology_data.get("logic_gates", []):
            gate = LogicGate(**gate_data)
            topology.logic_gates.append(gate)
        
        # Reconstruct regulatory interactions
        for reg_data in topology_data.get("regulatory_interactions", []):
            reg = RegulatoryInteraction(**reg_data)
            topology.regulatory_interactions.append(reg)
        
        data["topology"] = topology
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save circuit to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "Circuit":
        """Load circuit from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate the circuit and return error messages."""
        errors = []
        
        # Validate topology
        topology_errors = self.topology.validate()
        errors.extend(topology_errors)
        
        # Validate parameters
        if not isinstance(self.parameters, dict):
            errors.append("Parameters must be a dictionary")
        
        # Validate organism
        valid_organisms = ["E. coli", "S. cerevisiae", "B. subtilis", "P. putida", "mammalian"]
        if self.organism not in valid_organisms:
            errors.append(f"Organism must be one of: {valid_organisms}")
        
        return errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit statistics."""
        G = self.topology.to_networkx()
        
        return {
            "total_nodes": len(self.topology.nodes),
            "total_edges": len(self.topology.edges),
            "logic_gates": len(self.topology.logic_gates),
            "regulatory_interactions": len(self.topology.regulatory_interactions),
            "node_types": {node_type.value: len([n for n in self.topology.nodes if n.node_type == node_type]) 
                          for node_type in NodeType},
            "edge_types": {edge_type.value: len([e for e in self.topology.edges if e.edge_type == edge_type]) 
                          for edge_type in EdgeType},
            "is_connected": nx.is_weakly_connected(G) if len(G.nodes) > 1 else True,
            "has_cycles": len(list(nx.simple_cycles(G))) > 0 if len(G.nodes) > 0 else False,
            "diameter": nx.diameter(G) if nx.is_weakly_connected(G) and len(G.nodes) > 1 else 0,
            "average_clustering": nx.average_clustering(G) if len(G.nodes) > 2 else 0
        }
    
    def clone(self) -> "Circuit":
        """Create a deep copy of the circuit."""
        return Circuit.from_dict(self.to_dict()) 