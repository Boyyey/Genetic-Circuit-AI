"""
Biological Simulator Module for Genetic Circuit Design Platform.

This module provides advanced simulation capabilities for genetic circuits,
including ODE solvers, stochastic simulation, and parameter estimation.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, least_squares
# tellurium and roadrunner removed for Python 3.12 compatibility
# import tellurium as te
# import roadrunner
import pysb
from models.circuit import Circuit, CircuitNode, CircuitEdge
from models.simulation import SimulationParameters, SimulationResult, TimeSeries, ParameterSet


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ODESystem:
    """Represents a system of ordinary differential equations."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    variables: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "variables": self.variables,
            "equations": self.equations,
            "parameters": self.parameters,
            "initial_conditions": self.initial_conditions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ODESystem":
        """Create ODE system from dictionary."""
        return cls(**data)


class ODESolver:
    """Advanced ODE solver with multiple integration methods."""
    
    def __init__(self):
        """Initialize the ODE solver."""
        self.solvers = {
            "RK45": self._solve_rk45,
            "RK4": self._solve_rk4,
            "BDF": self._solve_bdf,
            "LSODA": self._solve_lsoda,
            "EULER": self._solve_euler,
            "HEUN": self._solve_heun
        }
        
        logger.info("ODE solver initialized successfully")
    
    def solve(self, ode_system: ODESystem, time_points: np.ndarray, 
              solver_type: str = "RK45", **kwargs) -> TimeSeries:
        """Solve ODE system and return time series."""
        logger.info(f"Solving ODE system with {solver_type} solver")
        
        if solver_type not in self.solvers:
            logger.warning(f"Unknown solver type: {solver_type}, using RK45")
            solver_type = "RK45"
        
        try:
            # Generate ODE function
            ode_func = self._generate_ode_function(ode_system)
            
            # Get initial conditions
            y0 = [ode_system.initial_conditions.get(var, 0.0) for var in ode_system.variables]
            
            # Solve ODE
            solution = self.solvers[solver_type](ode_func, y0, time_points, **kwargs)
            
            # Create time series
            time_series = TimeSeries(
                time_points=time_points,
                metadata={"solver": solver_type, "ode_system_id": ode_system.id}
            )
            
            # Add variables to time series
            for i, variable in enumerate(ode_system.variables):
                time_series.add_variable(variable, solution[:, i])
            
            logger.info("ODE system solved successfully")
            return time_series
            
        except Exception as e:
            logger.error(f"Error solving ODE system: {e}")
            raise
    
    def _generate_ode_function(self, ode_system: ODESystem) -> Callable:
        """Generate ODE function from system definition."""
        def ode_func(t, y):
            # Create variable dictionary
            variables = dict(zip(ode_system.variables, y))
            variables['t'] = t
            
            # Add parameters
            variables.update(ode_system.parameters)
            
            # Evaluate equations
            derivatives = []
            for equation in ode_system.equations:
                try:
                    # Simple evaluation - should be replaced with safe eval or symbolic computation
                    derivative = eval(equation, {"__builtins__": {}}, variables)
                    derivatives.append(derivative)
                except Exception as e:
                    logger.warning(f"Error evaluating equation {equation}: {e}")
                    derivatives.append(0.0)
            
            return derivatives
        
        return ode_func
    
    def _solve_rk45(self, ode_func: Callable, y0: List[float], 
                   time_points: np.ndarray, **kwargs) -> np.ndarray:
        """Solve using RK45 method."""
        solution = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            y0,
            method='RK45',
            t_eval=time_points,
            **kwargs
        )
        return solution.y.T
    
    def _solve_rk4(self, ode_func: Callable, y0: List[float], 
                  time_points: np.ndarray, **kwargs) -> np.ndarray:
        """Solve using RK4 method."""
        solution = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            y0,
            method='RK4',
            t_eval=time_points,
            **kwargs
        )
        return solution.y.T
    
    def _solve_bdf(self, ode_func: Callable, y0: List[float], 
                  time_points: np.ndarray, **kwargs) -> np.ndarray:
        """Solve using BDF method."""
        solution = solve_ivp(
            ode_func,
            (time_points[0], time_points[-1]),
            y0,
            method='BDF',
            t_eval=time_points,
            **kwargs
        )
        return solution.y.T
    
    def _solve_lsoda(self, ode_func: Callable, y0: List[float], 
                    time_points: np.ndarray, **kwargs) -> np.ndarray:
        """Solve using LSODA method."""
        solution = odeint(
            ode_func,
            y0,
            time_points,
            **kwargs
        )
        return solution
    
    def _solve_euler(self, ode_func: Callable, y0: List[float], 
                    time_points: np.ndarray, **kwargs) -> np.ndarray:
        """Solve using Euler method."""
        dt = time_points[1] - time_points[0]
        solution = np.zeros((len(time_points), len(y0)))
        solution[0] = y0
        
        for i in range(1, len(time_points)):
            derivatives = ode_func(time_points[i-1], solution[i-1])
            solution[i] = solution[i-1] + dt * np.array(derivatives)
        
        return solution
    
    def _solve_heun(self, ode_func: Callable, y0: List[float], 
                   time_points: np.ndarray, **kwargs) -> np.ndarray:
        """Solve using Heun method."""
        dt = time_points[1] - time_points[0]
        solution = np.zeros((len(time_points), len(y0)))
        solution[0] = y0
        
        for i in range(1, len(time_points)):
            k1 = np.array(ode_func(time_points[i-1], solution[i-1]))
            k2 = np.array(ode_func(time_points[i], solution[i-1] + dt * k1))
            solution[i] = solution[i-1] + 0.5 * dt * (k1 + k2)
        
        return solution


class BiologicalSimulator:
    """Main biological simulator for genetic circuits."""
    
    def __init__(self):
        """Initialize the biological simulator."""
        self.ode_solver = ODESolver()
        self.simulation_templates = self._load_simulation_templates()
        
        logger.info("Biological simulator initialized successfully")
    
    def _load_simulation_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load simulation templates for common circuit types."""
        return {
            "simple_repression": {
                "variables": ["mRNA", "Protein", "Repressor"],
                "equations": [
                    "transcription_rate * (1 - Repressor / (kd + Repressor)) - degradation_rate * mRNA",
                    "translation_rate * mRNA - protein_degradation_rate * Protein",
                    "repressor_production_rate - repressor_degradation_rate * Repressor"
                ],
                "parameters": {
                    "transcription_rate": 1.0,
                    "translation_rate": 1.0,
                    "degradation_rate": 0.1,
                    "protein_degradation_rate": 0.1,
                    "repressor_production_rate": 0.5,
                    "repressor_degradation_rate": 0.1,
                    "kd": 1.0
                },
                "initial_conditions": {
                    "mRNA": 0.0,
                    "Protein": 0.0,
                    "Repressor": 1.0
                }
            },
            "simple_activation": {
                "variables": ["mRNA", "Protein", "Activator"],
                "equations": [
                    "transcription_rate * (Activator / (kd + Activator)) - degradation_rate * mRNA",
                    "translation_rate * mRNA - protein_degradation_rate * Protein",
                    "activator_production_rate - activator_degradation_rate * Activator"
                ],
                "parameters": {
                    "transcription_rate": 1.0,
                    "translation_rate": 1.0,
                    "degradation_rate": 0.1,
                    "protein_degradation_rate": 0.1,
                    "activator_production_rate": 0.5,
                    "activator_degradation_rate": 0.1,
                    "kd": 1.0
                },
                "initial_conditions": {
                    "mRNA": 0.0,
                    "Protein": 0.0,
                    "Activator": 0.0
                }
            },
            "oscillator": {
                "variables": ["mRNA1", "Protein1", "mRNA2", "Protein2"],
                "equations": [
                    "transcription_rate1 * (1 - Protein2 / (kd1 + Protein2)) - degradation_rate1 * mRNA1",
                    "translation_rate1 * mRNA1 - protein_degradation_rate1 * Protein1",
                    "transcription_rate2 * (1 - Protein1 / (kd2 + Protein1)) - degradation_rate2 * mRNA2",
                    "translation_rate2 * mRNA2 - protein_degradation_rate2 * Protein2"
                ],
                "parameters": {
                    "transcription_rate1": 1.0,
                    "transcription_rate2": 1.0,
                    "translation_rate1": 1.0,
                    "translation_rate2": 1.0,
                    "degradation_rate1": 0.2,
                    "degradation_rate2": 0.2,
                    "protein_degradation_rate1": 0.2,
                    "protein_degradation_rate2": 0.2,
                    "kd1": 1.0,
                    "kd2": 1.0
                },
                "initial_conditions": {
                    "mRNA1": 0.1,
                    "Protein1": 0.0,
                    "mRNA2": 0.0,
                    "Protein2": 0.1
                }
            }
        }
    
    def simulate_circuit(self, circuit: Circuit, parameters: SimulationParameters) -> SimulationResult:
        """Simulate a genetic circuit."""
        logger.info(f"Starting simulation for circuit: {circuit.name}")
        
        try:
            # Generate ODE system from circuit
            ode_system = self._circuit_to_ode_system(circuit, parameters)
            
            # Get time points
            time_points = parameters.get_time_points()
            
            # Solve ODE system
            time_series = self.ode_solver.solve(
                ode_system,
                time_points,
                solver_type=parameters.solver_type.value,
                rtol=parameters.tolerance,
                max_step=parameters.time_step
            )
            
            # Create simulation result
            result = SimulationResult(
                simulation_type=parameters.simulation_type,
                time_series=time_series,
                parameters=parameters.parameters,
                metadata={
                    "circuit_id": circuit.id,
                    "circuit_name": circuit.name,
                    "simulation_parameters": parameters.to_dict()
                }
            )
            
            # Calculate performance metrics
            result.performance_metrics = result.calculate_metrics()
            
            logger.info("Simulation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    def _circuit_to_ode_system(self, circuit: Circuit, parameters: SimulationParameters) -> ODESystem:
        """Convert circuit to ODE system."""
        # Identify circuit template
        template = self._identify_circuit_template(circuit)
        
        # Get template ODE system
        template_system = self.simulation_templates.get(template, self.simulation_templates["simple_repression"])
        
        # Create ODE system
        ode_system = ODESystem(
            variables=template_system["variables"],
            equations=template_system["equations"],
            parameters=template_system["parameters"].copy(),
            initial_conditions=template_system["initial_conditions"].copy()
        )
        
        # Update parameters with circuit parameters
        ode_system.parameters.update(circuit.parameters)
        
        # Update parameters with simulation parameters
        ode_system.parameters.update(parameters.parameters.parameters)
        
        # Update initial conditions
        ode_system.initial_conditions.update(parameters.initial_conditions)
        
        return ode_system
    
    def _identify_circuit_template(self, circuit: Circuit) -> str:
        """Identify the simulation template for a circuit."""
        # Count components by type
        component_counts = {}
        for node in circuit.topology.nodes:
            node_type = node.node_type.value
            component_counts[node_type] = component_counts.get(node_type, 0) + 1
        
        # Identify template based on component composition
        if component_counts.get("regulator", 0) > 0 and component_counts.get("gene", 0) > 0:
            if self._has_feedback_loop(circuit):
                return "oscillator"
            else:
                return "simple_repression"
        elif component_counts.get("gene", 0) > 1:
            return "oscillator"
        else:
            return "simple_activation"
    
    def _has_feedback_loop(self, circuit: Circuit) -> bool:
        """Check if circuit has feedback loops."""
        # This function relies on networkx, which is not imported.
        # Assuming a placeholder or that networkx will be added later.
        # For now, returning False as a placeholder.
        return False
    
    def simulate_with_noise(self, circuit: Circuit, parameters: SimulationParameters, 
                          noise_level: float = 0.1) -> SimulationResult:
        """Simulate circuit with stochastic noise."""
        logger.info(f"Starting stochastic simulation for circuit: {circuit.name}")
        
        # Run deterministic simulation
        result = self.simulate_circuit(circuit, parameters)
        
        # Add noise to results
        noisy_time_series = TimeSeries(
            time_points=result.time_series.time_points,
            metadata=result.time_series.metadata
        )
        
        for variable, values in result.time_series.values.items():
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level * np.std(values), len(values))
            noisy_values = values + noise
            noisy_values = np.maximum(noisy_values, 0)  # Ensure non-negative
            noisy_time_series.add_variable(variable, noisy_values)
        
        # Create new result with noise
        noisy_result = SimulationResult(
            simulation_type=result.simulation_type,
            time_series=noisy_time_series,
            parameters=result.parameters,
            metadata=result.metadata
        )
        
        return noisy_result
    
    def parameter_estimation(self, circuit: Circuit, experimental_data: TimeSeries, 
                           parameters_to_estimate: List[str]) -> Dict[str, float]:
        """Estimate parameters from experimental data."""
        logger.info("Starting parameter estimation")
        
        def objective_function(params):
            # Update circuit parameters
            for param, value in zip(parameters_to_estimate, params):
                circuit.parameters[param] = value
            
            # Run simulation
            sim_params = SimulationParameters(
                time_end=experimental_data.time_points[-1],
                time_step=experimental_data.time_points[1] - experimental_data.time_points[0]
            )
            
            try:
                result = self.simulate_circuit(circuit, sim_params)
                
                # Calculate error
                error = 0
                for variable in experimental_data.values.keys():
                    if variable in result.time_series.values:
                        exp_values = experimental_data.values[variable]
                        sim_values = result.time_series.values[variable]
                        error += np.sum((exp_values - sim_values) ** 2)
                
                return error
                
            except Exception as e:
                logger.warning(f"Simulation failed during parameter estimation: {e}")
                return np.inf
        
        # Initial parameter values
        initial_params = [circuit.parameters.get(param, 1.0) for param in parameters_to_estimate]
        
        # Parameter bounds
        bounds = [(0.1, 10.0)] * len(parameters_to_estimate)
        
        # Optimize parameters
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if result.success:
            estimated_params = dict(zip(parameters_to_estimate, result.x))
            logger.info("Parameter estimation completed successfully")
            return estimated_params
        else:
            logger.warning("Parameter estimation failed")
            return dict(zip(parameters_to_estimate, initial_params))
    
    def sensitivity_analysis(self, circuit: Circuit, parameters: SimulationParameters,
                           parameter_names: List[str], perturbation: float = 0.1) -> Dict[str, float]:
        """Perform sensitivity analysis on circuit parameters."""
        logger.info("Starting sensitivity analysis")
        
        # Run baseline simulation
        baseline_result = self.simulate_circuit(circuit, parameters)
        baseline_output = baseline_result.get_final_values()
        
        sensitivities = {}
        
        for param_name in parameter_names:
            if param_name in circuit.parameters:
                # Store original value
                original_value = circuit.parameters[param_name]
                
                # Perturb parameter
                perturbed_value = original_value * (1 + perturbation)
                circuit.parameters[param_name] = perturbed_value
                
                # Run perturbed simulation
                perturbed_result = self.simulate_circuit(circuit, parameters)
                perturbed_output = perturbed_result.get_final_values()
                
                # Calculate sensitivity
                sensitivity = 0
                for variable in baseline_output.keys():
                    if variable in perturbed_output:
                        baseline_val = baseline_output[variable]
                        perturbed_val = perturbed_output[variable]
                        if baseline_val > 0:
                            sensitivity += abs((perturbed_val - baseline_val) / baseline_val) / perturbation
                
                sensitivities[param_name] = sensitivity
                
                # Restore original value
                circuit.parameters[param_name] = original_value
        
        logger.info("Sensitivity analysis completed")
        return sensitivities
    
    def monte_carlo_simulation(self, circuit: Circuit, parameters: SimulationParameters,
                             n_samples: int = 100, parameter_uncertainty: float = 0.2) -> List[SimulationResult]:
        """Run Monte Carlo simulation with parameter uncertainty."""
        logger.info(f"Starting Monte Carlo simulation with {n_samples} samples")
        
        results = []
        
        for i in range(n_samples):
            # Perturb parameters
            perturbed_circuit = circuit.clone()
            for param_name in perturbed_circuit.parameters:
                original_value = perturbed_circuit.parameters[param_name]
                perturbation = np.random.normal(0, parameter_uncertainty)
                perturbed_circuit.parameters[param_name] = original_value * (1 + perturbation)
            
            # Run simulation
            try:
                result = self.simulate_circuit(perturbed_circuit, parameters)
                results.append(result)
            except Exception as e:
                logger.warning(f"Monte Carlo sample {i} failed: {e}")
        
        logger.info(f"Monte Carlo simulation completed with {len(results)} successful samples")
        return results 