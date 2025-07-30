"""
Core module for the Genetic Circuit Design Platform.

This module contains the main functionality for:
- Natural language processing of genetic circuit descriptions
- Circuit design and optimization algorithms
- Biological simulation engines
- Visualization and analysis tools
- Genetic part database management

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

from .nlp_parser import NLPParser, CircuitLogicExtractor
from .circuit_designer import CircuitDesigner, CircuitOptimizer
from .simulator import BiologicalSimulator, ODESolver
from .visualizer import CircuitVisualizer, PlotGenerator
from .part_database import PartDatabase, PartManager

__version__ = "1.0.0"
__author__ = "Genetic Circuit Design Team"
__email__ = "support@geneticcircuit.design"

__all__ = [
    "NLPParser",
    "CircuitLogicExtractor", 
    "CircuitDesigner",
    "CircuitOptimizer",
    "BiologicalSimulator",
    "ODESolver",
    "CircuitVisualizer",
    "PlotGenerator",
    "PartDatabase",
    "PartManager"
] 