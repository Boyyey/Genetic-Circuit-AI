"""
Optimization utilities for genetic circuit design.

This module provides various optimization algorithms for parameter tuning,
circuit design optimization, and multi-objective optimization.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.stats import norm
import json

logger = logging.getLogger(__name__)


def genetic_algorithm(objective_func: Callable, bounds: List[Tuple[float, float]], 
                     population_size: int = 50, generations: int = 100,
                     mutation_rate: float = 0.1, crossover_rate: float = 0.8) -> Dict[str, Any]:
    """
    Genetic algorithm for optimization.
    
    Args:
        objective_func: Objective function to minimize
        bounds: Parameter bounds [(min, max), ...]
        population_size: Size of the population
        generations: Number of generations
        mutation_rate: Mutation probability
        crossover_rate: Crossover probability
        
    Returns:
        Optimization results
    """
    logger.info("Starting genetic algorithm optimization")
    
    # Initialize population
    population = []
    for _ in range(population_size):
        individual = [random.uniform(bound[0], bound[1]) for bound in bounds]
        population.append(individual)
    
    best_fitness = float('inf')
    best_individual = None
    fitness_history = []
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            try:
                fitness = objective_func(individual)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            except Exception as e:
                logger.warning(f"Error evaluating individual: {e}")
                fitness_scores.append(float('inf'))
        
        fitness_history.append(best_fitness)
        
        # Selection
        new_population = []
        
        # Elitism: keep best individual
        best_idx = np.argmin(fitness_scores)
        new_population.append(population[best_idx])
        
        # Tournament selection and reproduction
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parent1 = population[winner_idx]
            
            # Second parent
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parent2 = population[winner_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child, bounds)
            
            new_population.append(child)
        
        population = new_population
        
        if generation % 10 == 0:
            logger.info(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
    
    return {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
        "generations": generations,
        "population_size": population_size
    }


def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Perform crossover between two parents."""
    child = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child


def mutate(individual: List[float], bounds: List[Tuple[float, float]], 
          mutation_strength: float = 0.1) -> List[float]:
    """Mutate an individual."""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < 0.1:  # 10% chance of mutation per gene
            # Gaussian mutation
            mutation = np.random.normal(0, mutation_strength)
            mutated[i] += mutation
            
            # Ensure bounds
            mutated[i] = max(bounds[i][0], min(bounds[i][1], mutated[i]))
    
    return mutated


def particle_swarm_optimization(objective_func: Callable, bounds: List[Tuple[float, float]],
                              n_particles: int = 30, max_iterations: int = 100,
                              w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Dict[str, Any]:
    """
    Particle Swarm Optimization algorithm.
    
    Args:
        objective_func: Objective function to minimize
        bounds: Parameter bounds [(min, max), ...]
        n_particles: Number of particles
        max_iterations: Maximum iterations
        w: Inertia weight
        c1: Cognitive parameter
        c2: Social parameter
        
    Returns:
        Optimization results
    """
    logger.info("Starting particle swarm optimization")
    
    # Initialize particles
    particles = []
    velocities = []
    personal_best = []
    personal_best_fitness = []
    
    for _ in range(n_particles):
        # Random position
        position = [random.uniform(bound[0], bound[1]) for bound in bounds]
        particles.append(position)
        
        # Random velocity
        velocity = [random.uniform(-1, 1) for _ in bounds]
        velocities.append(velocity)
        
        # Evaluate fitness
        try:
            fitness = objective_func(position)
            personal_best.append(position.copy())
            personal_best_fitness.append(fitness)
        except Exception as e:
            logger.warning(f"Error evaluating particle: {e}")
            personal_best.append(position.copy())
            personal_best_fitness.append(float('inf'))
    
    # Global best
    global_best_idx = np.argmin(personal_best_fitness)
    global_best = personal_best[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]
    
    fitness_history = [global_best_fitness]
    
    for iteration in range(max_iterations):
        for i in range(n_particles):
            # Update velocity
            for j in range(len(bounds)):
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = c1 * r1 * (personal_best[i][j] - particles[i][j])
                social_velocity = c2 * r2 * (global_best[j] - particles[i][j])
                
                velocities[i][j] = w * velocities[i][j] + cognitive_velocity + social_velocity
            
            # Update position
            for j in range(len(bounds)):
                particles[i][j] += velocities[i][j]
                
                # Ensure bounds
                particles[i][j] = max(bounds[j][0], min(bounds[j][1], particles[i][j]))
            
            # Evaluate new position
            try:
                fitness = objective_func(particles[i])
                
                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness < global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                        
            except Exception as e:
                logger.warning(f"Error evaluating particle: {e}")
        
        fitness_history.append(global_best_fitness)
        
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: Best fitness = {global_best_fitness:.6f}")
    
    return {
        "best_individual": global_best,
        "best_fitness": global_best_fitness,
        "fitness_history": fitness_history,
        "iterations": max_iterations,
        "n_particles": n_particles
    }


def simulated_annealing(objective_func: Callable, bounds: List[Tuple[float, float]],
                       initial_temp: float = 100.0, final_temp: float = 1e-6,
                       max_iterations: int = 1000, cooling_rate: float = 0.95) -> Dict[str, Any]:
    """
    Simulated Annealing algorithm.
    
    Args:
        objective_func: Objective function to minimize
        bounds: Parameter bounds [(min, max), ...]
        initial_temp: Initial temperature
        final_temp: Final temperature
        max_iterations: Maximum iterations
        cooling_rate: Cooling rate
        
    Returns:
        Optimization results
    """
    logger.info("Starting simulated annealing optimization")
    
    # Initial solution
    current_solution = [random.uniform(bound[0], bound[1]) for bound in bounds]
    try:
        current_fitness = objective_func(current_solution)
    except Exception as e:
        logger.warning(f"Error evaluating initial solution: {e}")
        current_fitness = float('inf')
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    temperature = initial_temp
    fitness_history = [best_fitness]
    
    for iteration in range(max_iterations):
        # Generate neighbor
        neighbor = current_solution.copy()
        for i in range(len(neighbor)):
            # Gaussian perturbation
            perturbation = np.random.normal(0, 0.1)
            neighbor[i] += perturbation
            
            # Ensure bounds
            neighbor[i] = max(bounds[i][0], min(bounds[i][1], neighbor[i]))
        
        # Evaluate neighbor
        try:
            neighbor_fitness = objective_func(neighbor)
        except Exception as e:
            logger.warning(f"Error evaluating neighbor: {e}")
            neighbor_fitness = float('inf')
        
        # Accept or reject
        delta_e = neighbor_fitness - current_fitness
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / temperature):
            current_solution = neighbor
            current_fitness = neighbor_fitness
            
            # Update best solution
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Cool down
        temperature *= cooling_rate
        
        if temperature < final_temp:
            break
        
        fitness_history.append(best_fitness)
        
        if iteration % 100 == 0:
            logger.info(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}, Temp = {temperature:.6f}")
    
    return {
        "best_individual": best_solution,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
        "iterations": iteration + 1,
        "final_temperature": temperature
    }


def bayesian_optimization(objective_func: Callable, bounds: List[Tuple[float, float]],
                         n_initial_points: int = 5, n_iterations: int = 50,
                         acquisition_function: str = "ei") -> Dict[str, Any]:
    """
    Bayesian Optimization using Gaussian Process.
    
    Args:
        objective_func: Objective function to minimize
        bounds: Parameter bounds [(min, max), ...]
        n_initial_points: Number of initial random points
        n_iterations: Number of optimization iterations
        acquisition_function: Acquisition function type ("ei", "ucb", "pi")
        
    Returns:
        Optimization results
    """
    logger.info("Starting Bayesian optimization")
    
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    except ImportError:
        logger.error("scikit-learn is required for Bayesian optimization")
        return {"error": "scikit-learn not available"}
    
    # Initialize data
    X = []
    y = []
    
    # Initial random points
    for _ in range(n_initial_points):
        point = [random.uniform(bound[0], bound[1]) for bound in bounds]
        try:
            value = objective_func(point)
            X.append(point)
            y.append(value)
        except Exception as e:
            logger.warning(f"Error evaluating initial point: {e}")
    
    if not X:
        logger.error("No valid initial points")
        return {"error": "No valid initial points"}
    
    X = np.array(X)
    y = np.array(y)
    
    # Gaussian Process
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    
    best_fitness = min(y)
    best_individual = X[np.argmin(y)]
    fitness_history = [best_fitness]
    
    for iteration in range(n_iterations):
        # Fit GP
        gp.fit(X, y)
        
        # Generate candidate points
        n_candidates = 100
        candidates = []
        for _ in range(n_candidates):
            candidate = [random.uniform(bound[0], bound[1]) for bound in bounds]
            candidates.append(candidate)
        
        candidates = np.array(candidates)
        
        # Predict mean and std
        mean, std = gp.predict(candidates, return_std=True)
        
        # Acquisition function
        if acquisition_function == "ei":
            # Expected Improvement
            best_so_far = min(y)
            improvement = best_so_far - mean
            z = improvement / (std + 1e-9)
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0] = 0
            next_point_idx = np.argmax(ei)
            
        elif acquisition_function == "ucb":
            # Upper Confidence Bound
            ucb = mean + 2 * std
            next_point_idx = np.argmin(ucb)
            
        elif acquisition_function == "pi":
            # Probability of Improvement
            best_so_far = min(y)
            z = (best_so_far - mean) / (std + 1e-9)
            pi = norm.cdf(z)
            next_point_idx = np.argmax(pi)
            
        else:
            next_point_idx = np.argmin(mean)
        
        # Evaluate next point
        next_point = candidates[next_point_idx]
        try:
            next_value = objective_func(next_point)
            X = np.vstack([X, next_point])
            y = np.append(y, next_value)
            
            # Update best
            if next_value < best_fitness:
                best_fitness = next_value
                best_individual = next_point.copy()
                
        except Exception as e:
            logger.warning(f"Error evaluating next point: {e}")
        
        fitness_history.append(best_fitness)
        
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
    
    return {
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
        "iterations": n_iterations,
        "acquisition_function": acquisition_function
    }


def multi_objective_optimization(objective_functions: List[Callable], bounds: List[Tuple[float, float]],
                               algorithm: str = "nsga2", population_size: int = 100,
                               generations: int = 100) -> Dict[str, Any]:
    """
    Multi-objective optimization.
    
    Args:
        objective_functions: List of objective functions to minimize
        bounds: Parameter bounds [(min, max), ...]
        algorithm: Optimization algorithm ("nsga2", "pareto")
        population_size: Population size
        generations: Number of generations
        
    Returns:
        Optimization results
    """
    logger.info(f"Starting multi-objective optimization with {algorithm}")
    
    if algorithm == "nsga2":
        return _nsga2_optimization(objective_functions, bounds, population_size, generations)
    elif algorithm == "pareto":
        return _pareto_optimization(objective_functions, bounds, population_size, generations)
    else:
        logger.error(f"Unknown multi-objective algorithm: {algorithm}")
        return {"error": f"Unknown algorithm: {algorithm}"}


def _nsga2_optimization(objective_functions: List[Callable], bounds: List[Tuple[float, float]],
                       population_size: int, generations: int) -> Dict[str, Any]:
    """NSGA-II multi-objective optimization."""
    # Simplified NSGA-II implementation
    population = []
    
    # Initialize population
    for _ in range(population_size):
        individual = [random.uniform(bound[0], bound[1]) for bound in bounds]
        population.append(individual)
    
    pareto_front = []
    
    for generation in range(generations):
        # Evaluate objectives
        objectives = []
        for individual in population:
            try:
                obj_values = [func(individual) for func in objective_functions]
                objectives.append(obj_values)
            except Exception as e:
                logger.warning(f"Error evaluating individual: {e}")
                objectives.append([float('inf')] * len(objective_functions))
        
        # Find Pareto front
        pareto_indices = _find_pareto_front(objectives)
        pareto_front = [population[i] for i in pareto_indices]
        
        # Selection and reproduction (simplified)
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child, bounds)
            new_population.append(child)
        
        population = new_population
        
        if generation % 10 == 0:
            logger.info(f"Generation {generation}: Pareto front size = {len(pareto_front)}")
    
    return {
        "pareto_front": pareto_front,
        "generations": generations,
        "population_size": population_size
    }


def _pareto_optimization(objective_functions: List[Callable], bounds: List[Tuple[float, float]],
                        population_size: int, generations: int) -> Dict[str, Any]:
    """Pareto-based multi-objective optimization."""
    # Simplified Pareto optimization
    return _nsga2_optimization(objective_functions, bounds, population_size, generations)


def _find_pareto_front(objectives: List[List[float]]) -> List[int]:
    """Find Pareto front indices."""
    pareto_indices = []
    
    for i, obj1 in enumerate(objectives):
        is_dominated = False
        
        for j, obj2 in enumerate(objectives):
            if i != j:
                # Check if obj2 dominates obj1
                dominates = True
                for k in range(len(obj1)):
                    if obj2[k] > obj1[k]:
                        dominates = False
                        break
                
                if dominates:
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices


def optimize_circuit_parameters(circuit, objective_function: Callable,
                              parameter_names: List[str], bounds: List[Tuple[float, float]],
                              algorithm: str = "genetic_algorithm", **kwargs) -> Dict[str, Any]:
    """
    Optimize circuit parameters.
    
    Args:
        circuit: Circuit object to optimize
        objective_function: Objective function
        parameter_names: Names of parameters to optimize
        bounds: Parameter bounds
        algorithm: Optimization algorithm
        **kwargs: Additional algorithm parameters
        
    Returns:
        Optimization results
    """
    logger.info(f"Starting circuit parameter optimization with {algorithm}")
    
    def wrapped_objective(params):
        # Update circuit parameters
        for name, value in zip(parameter_names, params):
            circuit.parameters[name] = value
        
        # Evaluate objective
        try:
            return objective_function(circuit)
        except Exception as e:
            logger.warning(f"Error evaluating objective: {e}")
            return float('inf')
    
    # Run optimization
    if algorithm == "genetic_algorithm":
        return genetic_algorithm(wrapped_objective, bounds, **kwargs)
    elif algorithm == "particle_swarm":
        return particle_swarm_optimization(wrapped_objective, bounds, **kwargs)
    elif algorithm == "simulated_annealing":
        return simulated_annealing(wrapped_objective, bounds, **kwargs)
    elif algorithm == "bayesian":
        return bayesian_optimization(wrapped_objective, bounds, **kwargs)
    else:
        logger.error(f"Unknown optimization algorithm: {algorithm}")
        return {"error": f"Unknown algorithm: {algorithm}"} 