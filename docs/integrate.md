 
GET /health

Response:

{
  "status": "healthy",
  "timestamp": "2025-11-25T00:17:15.053513",
  "service": "greedyoptim-api"
}


GET /methods

Response:

 {
  "available_methods": {
    "ga": {
      "name": "Genetic Algorithm",
      "description": "Evolutionary optimization using selection, crossover, and mutation",
      "typical_time": "medium",
      "solution_quality": "high"
    },
    "cmaes": {
      "name": "CMA-ES",
      "description": "Covariance Matrix Adaptation Evolution Strategy",
      "typical_time": "medium-high",
      "solution_quality": "very high"
    },
    "pso": {
      "name": "Particle Swarm Optimization",
      "description": "Swarm intelligence-based optimization",
      "typical_time": "medium",
      "solution_quality": "high"
    },
    "sa": {
      "name": "Simulated Annealing",
      "description": "Probabilistic optimization inspired by metallurgy",
      "typical_time": "medium",
      "solution_quality": "medium-high"
    },
    "nsga2": {
      "name": "NSGA-II",
      "description": "Non-dominated Sorting Genetic Algorithm (multi-objective)",
      "typical_time": "high",
      "solution_quality": "very high"
    },
    "adaptive": {
      "name": "Adaptive Optimizer",
      "description": "Automatically selects best algorithm",
      "typical_time": "high",
      "solution_quality": "very high"
    },
    "ensemble": {
      "name": "Ensemble Optimizer",
      "description": "Runs multiple algorithms in parallel",
      "typical_time": "high",
      "solution_quality": "highest"
    }
  },
  "default_method": "ga",
  "recommended_for_speed": "ga",
  "recommended_for_quality": "ensemble"
}

