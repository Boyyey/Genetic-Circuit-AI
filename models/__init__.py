"""
Data models and schemas for the Genetic Circuit Design Platform.

This module defines the core data structures used throughout the application:
- Circuit representations and topologies
- Genetic part definitions and properties
- Simulation parameters and results
- Biological entity models

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

from .circuit import (
    Circuit,
    CircuitNode,
    CircuitEdge,
    CircuitTopology,
    LogicGate,
    RegulatoryInteraction
)
from .parts import (
    GeneticPart,
    Promoter,
    Gene,
    Terminator,
    RibosomeBindingSite,
    Protein,
    SmallMolecule
)
from .simulation import (
    SimulationParameters,
    SimulationResult,
    TimeSeries,
    ParameterSet,
    OptimizationResult
)

__version__ = "1.0.0"
__author__ = "Genetic Circuit Design Team"

__all__ = [
    # Circuit models
    "Circuit",
    "CircuitNode", 
    "CircuitEdge",
    "CircuitTopology",
    "LogicGate",
    "RegulatoryInteraction",
    
    # Part models
    "GeneticPart",
    "Promoter",
    "Gene", 
    "Terminator",
    "RibosomeBindingSite",
    "Protein",
    "SmallMolecule",
    
    # Simulation models
    "SimulationParameters",
    "SimulationResult",
    "TimeSeries",
    "ParameterSet",
    "OptimizationResult"
] 