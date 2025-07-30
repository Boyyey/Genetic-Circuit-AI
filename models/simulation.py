"""
Simulation models for biological system analysis.

This module defines data structures for simulation parameters, results,
time series data, and optimization results. It provides a comprehensive
framework for modeling and analyzing genetic circuit dynamics.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution


class SimulationType(Enum):
    """Types of simulations."""
    ODE = "ordinary_differential_equations"
    SDE = "stochastic_differential_equations"
    GILLESPIE = "gillespie"
    MONTE_CARLO = "monte_carlo"
    AGENT_BASED = "agent_based"
    HYBRID = "hybrid"


class SolverType(Enum):
    """Types of ODE solvers."""
    RK4 = "runge_kutta_4"
    RK45 = "runge_kutta_45"
    BDF = "backward_differentiation_formula"
    LSODA = "lsoda"
    EULER = "euler"
    HEUN = "heun"


class OptimizationMethod(Enum):
    """Types of optimization methods."""
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    BAYESIAN = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary_strategy"


@dataclass
class TimeSeries:
    """Represents time series data."""
    
    time_points: np.ndarray = field(default_factory=lambda: np.array([]))
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate time series data."""
        if len(self.time_points) == 0:
            return
        
        # Ensure all value arrays have the same length as time_points
        for key, values in self.values.items():
            if len(values) != len(self.time_points):
                raise ValueError(f"Time series {key} has length {len(values)}, expected {len(self.time_points)}")
    
    def add_variable(self, name: str, values: np.ndarray) -> None:
        """Add a new variable to the time series."""
        if len(self.time_points) > 0 and len(values) != len(self.time_points):
            raise ValueError(f"Variable {name} has length {len(values)}, expected {len(self.time_points)}")
        self.values[name] = values
    
    def get_variable(self, name: str) -> Optional[np.ndarray]:
        """Get a variable by name."""
        return self.values.get(name)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {"time": self.time_points}
        data.update(self.values)
        return pd.DataFrame(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "time_points": self.time_points.tolist(),
            "values": {k: v.tolist() for k, v in self.values.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeries":
        """Create time series from dictionary."""
        return cls(
            time_points=np.array(data["time_points"]),
            values={k: np.array(v) for k, v in data["values"].items()},
            metadata=data["metadata"]
        )
    
    def plot(self, variables: Optional[List[str]] = None, **kwargs) -> Any:
        """Plot the time series data."""
        try:
            import matplotlib.pyplot as plt
            
            if variables is None:
                variables = list(self.values.keys())
            
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
            
            for var in variables:
                if var in self.values:
                    ax.plot(self.time_points, self.values[var], label=var, **kwargs)
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            raise ImportError("matplotlib is required for plotting")


@dataclass
class ParameterSet:
    """Represents a set of simulation parameters."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    parameters: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate parameter set."""
        if not self.name:
            self.name = f"ParameterSet_{self.id[:8]}"
    
    def get_parameter(self, name: str, default: float = 0.0) -> float:
        """Get a parameter value."""
        return self.parameters.get(name, default)
    
    def set_parameter(self, name: str, value: float) -> None:
        """Set a parameter value."""
        self.parameters[name] = value
    
    def get_bounds(self, name: str) -> Tuple[float, float]:
        """Get parameter bounds."""
        return self.bounds.get(name, (0.0, float('inf')))
    
    def set_bounds(self, name: str, lower: float, upper: float) -> None:
        """Set parameter bounds."""
        self.bounds[name] = (lower, upper)
    
    def validate(self) -> List[str]:
        """Validate parameter set and return error messages."""
        errors = []
        
        for param_name, value in self.parameters.items():
            if param_name in self.bounds:
                lower, upper = self.bounds[param_name]
                if value < lower or value > upper:
                    errors.append(f"Parameter {param_name} value {value} outside bounds [{lower}, {upper}]")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "parameters": self.parameters,
            "bounds": self.bounds,
            "units": self.units,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSet":
        """Create parameter set from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class SimulationParameters(BaseModel):
    """Parameters for biological simulation."""
    
    simulation_type: SimulationType = SimulationType.ODE
    solver_type: SolverType = SolverType.RK45
    time_start: float = 0.0
    time_end: float = 100.0
    time_step: float = 0.1
    tolerance: float = 1e-6
    max_steps: int = 10000
    parameters: ParameterSet = Field(default_factory=ParameterSet)
    initial_conditions: Dict[str, float] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    noise_parameters: Dict[str, float] = Field(default_factory=dict)
    
    @validator("time_end")
    def validate_time_end(cls, v, values):
        """Validate time end is greater than time start."""
        if "time_start" in values and v <= values["time_start"]:
            raise ValueError("time_end must be greater than time_start")
        return v
    
    @validator("time_step")
    def validate_time_step(cls, v, values):
        """Validate time step is positive and reasonable."""
        if v <= 0:
            raise ValueError("time_step must be positive")
        if "time_end" in values and "time_start" in values:
            total_time = values["time_end"] - values["time_start"]
            if v > total_time / 10:
                raise ValueError("time_step should be smaller than 1/10 of total simulation time")
        return v
    
    def get_time_points(self) -> np.ndarray:
        """Get time points for simulation."""
        return np.arange(self.time_start, self.time_end + self.time_step, self.time_step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "simulation_type": self.simulation_type.value,
            "solver_type": self.solver_type.value,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "time_step": self.time_step,
            "tolerance": self.tolerance,
            "max_steps": self.max_steps,
            "parameters": self.parameters.to_dict(),
            "initial_conditions": self.initial_conditions,
            "events": self.events,
            "noise_parameters": self.noise_parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationParameters":
        """Create simulation parameters from dictionary."""
        data["simulation_type"] = SimulationType(data["simulation_type"])
        data["solver_type"] = SolverType(data["solver_type"])
        data["parameters"] = ParameterSet.from_dict(data["parameters"])
        return cls(**data)


@dataclass
class SimulationResult:
    """Results from a biological simulation."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    simulation_type: SimulationType = SimulationType.ODE
    time_series: TimeSeries = field(default_factory=TimeSeries)
    parameters: ParameterSet = field(default_factory=ParameterSet)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_final_values(self) -> Dict[str, float]:
        """Get final values of all variables."""
        if len(self.time_series.time_points) == 0:
            return {}
        
        return {name: values[-1] for name, values in self.time_series.values.items()}
    
    def get_steady_state(self, tolerance: float = 1e-6) -> Dict[str, float]:
        """Estimate steady state values."""
        if len(self.time_series.time_points) < 10:
            return self.get_final_values()
        
        # Use last 10% of simulation to estimate steady state
        start_idx = int(0.9 * len(self.time_series.time_points))
        steady_state = {}
        
        for name, values in self.time_series.values.items():
            final_values = values[start_idx:]
            if np.std(final_values) < tolerance:
                steady_state[name] = np.mean(final_values)
            else:
                steady_state[name] = values[-1]  # Use final value if not converged
        
        return steady_state
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        if len(self.time_series.time_points) == 0:
            return metrics
        
        # Calculate basic statistics for each variable
        for name, values in self.time_series.values.items():
            metrics[f"{name}_max"] = np.max(values)
            metrics[f"{name}_min"] = np.min(values)
            metrics[f"{name}_mean"] = np.mean(values)
            metrics[f"{name}_std"] = np.std(values)
            metrics[f"{name}_final"] = values[-1]
        
        # Calculate system-level metrics
        all_values = np.concatenate(list(self.time_series.values.values()))
        metrics["system_max"] = np.max(all_values)
        metrics["system_min"] = np.min(all_values)
        metrics["system_mean"] = np.mean(all_values)
        
        # Calculate convergence metrics
        steady_state = self.get_steady_state()
        metrics["convergence_error"] = np.mean([
            abs(values[-1] - steady_state.get(name, values[-1]))
            for name, values in self.time_series.values.items()
        ])
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "simulation_type": self.simulation_type.value,
            "time_series": self.time_series.to_dict(),
            "parameters": self.parameters.to_dict(),
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "convergence_info": self.convergence_info,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationResult":
        """Create simulation result from dictionary."""
        data["simulation_type"] = SimulationType(data["simulation_type"])
        data["time_series"] = TimeSeries.from_dict(data["time_series"])
        data["parameters"] = ParameterSet.from_dict(data["parameters"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    optimization_method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM
    objective_function: str = ""
    optimal_parameters: ParameterSet = field(default_factory=ParameterSet)
    optimal_value: float = float('inf')
    convergence_history: List[float] = field(default_factory=list)
    parameter_history: List[Dict[str, float]] = field(default_factory=list)
    iterations: int = 0
    success: bool = False
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_best_parameters(self) -> ParameterSet:
        """Get the best parameters found during optimization."""
        if self.parameter_history:
            # Find iteration with best objective value
            best_idx = np.argmin(self.convergence_history)
            best_params = self.parameter_history[best_idx]
            
            optimal_set = ParameterSet(
                name=f"Optimized_{self.id[:8]}",
                parameters=best_params
            )
            return optimal_set
        
        return self.optimal_parameters
    
    def plot_convergence(self, **kwargs) -> Any:
        """Plot optimization convergence."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
            ax.plot(self.convergence_history, 'b-', linewidth=2)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Objective Value")
            ax.set_title("Optimization Convergence")
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "optimization_method": self.optimization_method.value,
            "objective_function": self.objective_function,
            "optimal_parameters": self.optimal_parameters.to_dict(),
            "optimal_value": self.optimal_value,
            "convergence_history": self.convergence_history,
            "parameter_history": self.parameter_history,
            "iterations": self.iterations,
            "success": self.success,
            "message": self.message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Create optimization result from dictionary."""
        data["optimization_method"] = OptimizationMethod(data["optimization_method"])
        data["optimal_parameters"] = ParameterSet.from_dict(data["optimal_parameters"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class SensitivityAnalysis(BaseModel):
    """Sensitivity analysis results."""
    
    parameter_names: List[str] = Field(default_factory=list)
    sensitivity_matrix: np.ndarray = Field(default_factory=lambda: np.array([]))
    local_sensitivities: Dict[str, float] = Field(default_factory=dict)
    global_sensitivities: Dict[str, float] = Field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = Field(default_factory=dict)

    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def get_most_sensitive_parameters(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n most sensitive parameters."""
        if not self.global_sensitivities:
            return []
        
        sorted_params = sorted(
            self.global_sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_params[:n]
    
    def plot_sensitivity(self, **kwargs) -> Any:
        """Plot sensitivity analysis results."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.global_sensitivities:
                raise ValueError("No sensitivity data available")
            
            params = list(self.global_sensitivities.keys())
            sensitivities = list(self.global_sensitivities.values())
            
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
            bars = ax.bar(params, np.abs(sensitivities))
            ax.set_xlabel("Parameters")
            ax.set_ylabel("Sensitivity (absolute value)")
            ax.set_title("Parameter Sensitivity Analysis")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            return fig, ax
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parameter_names": self.parameter_names,
            "sensitivity_matrix": self.sensitivity_matrix.tolist(),
            "local_sensitivities": self.local_sensitivities,
            "global_sensitivities": self.global_sensitivities,
            "confidence_intervals": self.confidence_intervals
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensitivityAnalysis":
        """Create sensitivity analysis from dictionary."""
        data["sensitivity_matrix"] = np.array(data["sensitivity_matrix"])
        return cls(**data) 