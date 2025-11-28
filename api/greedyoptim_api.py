"""
FastAPI Service for GreedyOptim Scheduling
Exposes greedyOptim functionality with customizable input data
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import greedyOptim components
from greedyOptim.scheduler import optimize_trainset_schedule, compare_optimization_methods
from greedyOptim.models import OptimizationConfig, OptimizationResult
from greedyOptim.error_handling import DataValidator

# Import DataService for synthetic data generation (optional)
from DataService.enhanced_generator import EnhancedMetroDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GreedyOptim Scheduling API",
    description="Advanced train scheduling optimization using genetic algorithms, PSO, CMA-ES, and more",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class TrainsetStatusInput(BaseModel):
    """Single trainset operational status"""
    trainset_id: str
    operational_status: str = Field(..., description="IN_SERVICE, STANDBY, MAINTENANCE, OUT_OF_SERVICE, TESTING (or legacy: Available, In-Service, Maintenance, Standby, Out-of-Order)")
    last_maintenance_date: Optional[str] = None
    total_mileage_km: Optional[float] = None
    age_years: Optional[float] = None


class FitnessCertificateInput(BaseModel):
    """Fitness certificate for a trainset"""
    trainset_id: str
    department: str = Field(..., description="Safety, Operations, Technical, Electrical, Mechanical")
    status: str = Field(..., description="ISSUED, EXPIRED, SUSPENDED, PENDING, IN_PROGRESS, REVOKED, RENEWED, CANCELLED (or legacy: Valid, Expired, Expiring-Soon, Suspended)")
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None


class JobCardInput(BaseModel):
    """Job card/work order for trainset"""
    trainset_id: str
    job_id: str
    priority: str = Field(..., description="Critical, High, Medium, Low")
    status: str = Field(..., description="Open, In-Progress, Closed, Pending-Parts")
    description: Optional[str] = None
    estimated_hours: Optional[float] = None


class ComponentHealthInput(BaseModel):
    """Component health status"""
    trainset_id: str
    component: str = Field(..., description="Brakes, HVAC, Doors, Propulsion, etc.")
    status: str = Field(..., description="EXCELLENT, GOOD, FAIR, POOR, CRITICAL, FAILED (or legacy: Good, Fair, Warning, Critical)")
    wear_level: Optional[float] = Field(None, ge=0, le=100)
    last_inspection: Optional[str] = None


class OptimizationConfigInput(BaseModel):
    """Configuration for optimization algorithm"""
    required_service_trains: Optional[int] = Field(15, description="Minimum trains required in service")
    min_standby: Optional[int] = Field(2, description="Minimum standby trains")
    
    # Genetic Algorithm parameters
    population_size: Optional[int] = Field(50, ge=10, le=200)
    generations: Optional[int] = Field(100, ge=10, le=1000)
    mutation_rate: Optional[float] = Field(0.1, ge=0.0, le=1.0)
    crossover_rate: Optional[float] = Field(0.8, ge=0.0, le=1.0)
    elite_size: Optional[int] = Field(5, ge=1)


class ScheduleOptimizationRequest(BaseModel):
    """Request for schedule optimization"""
    trainset_status: List[TrainsetStatusInput]
    fitness_certificates: List[FitnessCertificateInput]
    job_cards: Optional[List[JobCardInput]] = Field(default_factory=list, description="Job cards are optional, defaults to empty list")
    component_health: List[ComponentHealthInput]
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None
    date: Optional[str] = Field(None, description="Date for schedule (YYYY-MM-DD)")
    
    # Optimization configuration
    config: Optional[OptimizationConfigInput] = None
    method: str = Field("ga", description="Optimization method: ga, cmaes, pso, sa, nsga2, adaptive, ensemble")
    
    # Optional additional data
    branding_contracts: Optional[List[Dict[str, Any]]] = None
    maintenance_schedule: Optional[List[Dict[str, Any]]] = None
    performance_metrics: Optional[List[Dict[str, Any]]] = None


class CompareMethodsRequest(BaseModel):
    """Request to compare multiple optimization methods"""
    trainset_status: List[TrainsetStatusInput]
    fitness_certificates: List[FitnessCertificateInput]
    job_cards: Optional[List[JobCardInput]] = Field(default_factory=list, description="Job cards are optional, defaults to empty list")
    component_health: List[ComponentHealthInput]
    
    metadata: Optional[Dict[str, Any]] = None
    date: Optional[str] = None
    config: Optional[OptimizationConfigInput] = None
    methods: List[str] = Field(["ga", "pso", "cmaes"], description="Methods to compare")


class SyntheticDataRequest(BaseModel):
    """Request to generate synthetic data"""
    num_trainsets: int = Field(25, ge=5, le=100, description="Number of trainsets to generate")
    maintenance_rate: float = Field(0.1, ge=0.0, le=0.5, description="Percentage in maintenance")
    availability_rate: float = Field(0.8, ge=0.5, le=1.0, description="Percentage available for service")


class ScheduleOptimizationResponse(BaseModel):
    """Response from optimization"""
    success: bool
    method: str
    fitness_score: float
    
    # Schedule allocation
    service_trains: List[str]
    standby_trains: List[str]
    maintenance_trains: List[str]
    unavailable_trains: List[str]
    
    # Metrics
    num_service: int
    num_standby: int
    num_maintenance: int
    num_unavailable: int
    
    # Detailed scores
    service_score: float
    standby_score: float
    health_score: float
    certificate_score: float
    
    # Metadata
    execution_time_seconds: Optional[float] = None
    timestamp: str
    constraints_satisfied: bool
    warnings: Optional[List[str]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def convert_pydantic_to_dict(request: ScheduleOptimizationRequest) -> Dict[str, Any]:
    """Convert Pydantic request model to dict format expected by greedyOptim"""
    data = {
        "trainset_status": [ts.dict() for ts in request.trainset_status],
        "fitness_certificates": [fc.dict() for fc in request.fitness_certificates],
        "job_cards": [jc.dict() for jc in request.job_cards] if request.job_cards else [],
        "component_health": [ch.dict() for ch in request.component_health],
        "metadata": request.metadata or {
            "generated_at": datetime.now().isoformat(),
            "system": "Kochi Metro Rail",
            "date": request.date or datetime.now().strftime("%Y-%m-%d")
        }
    }
    
    # Add optional data if provided
    if request.branding_contracts:
        data["branding_contracts"] = request.branding_contracts
    if request.maintenance_schedule:
        data["maintenance_schedule"] = request.maintenance_schedule
    if request.performance_metrics:
        data["performance_metrics"] = request.performance_metrics
    
    return data


def convert_config(config_input: Optional[OptimizationConfigInput]) -> OptimizationConfig:
    """Convert Pydantic config to OptimizationConfig"""
    if config_input is None:
        return OptimizationConfig()
    
    return OptimizationConfig(
        required_service_trains=config_input.required_service_trains or 15,
        min_standby=config_input.min_standby or 2,
        population_size=config_input.population_size or 50,
        generations=config_input.generations or 100,
        mutation_rate=config_input.mutation_rate or 0.1,
        crossover_rate=config_input.crossover_rate or 0.8,
        elite_size=config_input.elite_size or 5
    )


def convert_result_to_response(
    result: OptimizationResult,
    method: str,
    execution_time: Optional[float] = None
) -> ScheduleOptimizationResponse:
    """Convert OptimizationResult to API response"""
    # Extract objectives
    objectives = result.objectives
    
    # Determine unavailable trains (those not selected, standby, or maintenance)
    all_trains = set(result.selected_trainsets + result.standby_trainsets + result.maintenance_trainsets)
    unavailable = []  # We don't have this info in current result structure
    
    return ScheduleOptimizationResponse(
        success=True,
        method=method,
        fitness_score=result.fitness_score,
        service_trains=result.selected_trainsets,
        standby_trains=result.standby_trainsets,
        maintenance_trains=result.maintenance_trainsets,
        unavailable_trains=unavailable,
        num_service=len(result.selected_trainsets),
        num_standby=len(result.standby_trainsets),
        num_maintenance=len(result.maintenance_trainsets),
        num_unavailable=len(unavailable),
        service_score=objectives.get('service', 0.0),
        standby_score=objectives.get('standby', 0.0),
        health_score=objectives.get('health', 0.0),
        certificate_score=objectives.get('certificates', 0.0),
        execution_time_seconds=execution_time,
        timestamp=datetime.now().isoformat(),
        constraints_satisfied=len(result.selected_trainsets) >= 10,  # Basic check
        warnings=None
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "GreedyOptim Scheduling API",
        "version": "2.0.0",
        "description": "Advanced train scheduling optimization",
        "endpoints": {
            "POST /optimize": "Optimize schedule with custom data",
            "POST /compare": "Compare multiple optimization methods",
            "POST /generate-synthetic": "Generate synthetic test data",
            "POST /validate": "Validate input data structure",
            "GET /health": "Health check",
            "GET /methods": "List available optimization methods",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "greedyoptim-api"
    }


@app.get("/methods")
async def list_methods():
    """List available optimization methods"""
    return {
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


@app.post("/optimize", response_model=ScheduleOptimizationResponse)
async def optimize_schedule(request: ScheduleOptimizationRequest):
    """
    Optimize train schedule with custom input data.
    
    This endpoint accepts detailed trainset data and returns an optimized schedule
    that maximizes service coverage while respecting all constraints.
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Received optimization request with {len(request.trainset_status)} trainsets, method: {request.method}")
        
        # Convert request to dict format
        data = convert_pydantic_to_dict(request)
        
        # Validate data
        validation_errors = DataValidator.validate_data(data)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Data validation failed",
                    "validation_errors": validation_errors,
                    "message": "Please fix the data structure and try again"
                }
            )
        
        # Convert config
        config = convert_config(request.config)
        
        # Run optimization
        result = optimize_trainset_schedule(data, request.method, config)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {execution_time:.3f}s, fitness: {result.fitness_score:.4f}")
        
        # Convert to response
        response = convert_result_to_response(result, request.method, execution_time)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Optimization failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@app.post("/compare")
async def compare_methods(request: CompareMethodsRequest):
    """
    Compare multiple optimization methods on the same input data.
    
    Returns results from all requested methods for comparison.
    """
    try:
        import time
        
        logger.info(f"Comparing methods: {request.methods}")
        
        # Create a temporary request object for conversion
        temp_request = ScheduleOptimizationRequest(
            trainset_status=request.trainset_status,
            fitness_certificates=request.fitness_certificates,
            job_cards=request.job_cards,
            component_health=request.component_health,
            metadata=request.metadata,
            date=request.date,
            method="ga"  # Default method for conversion
        )
        
        # Convert request to dict format
        data = convert_pydantic_to_dict(temp_request)
        
        # Validate data
        validation_errors = DataValidator.validate_data(data)
        if validation_errors:
            raise HTTPException(status_code=400, detail={"error": "Data validation failed", "details": validation_errors})
        
        # Convert config
        config = convert_config(request.config)
        
        # Compare methods
        start_time = time.time()
        results = compare_optimization_methods(data, request.methods, config)
        total_time = time.time() - start_time
        
        # Convert results
        comparison = {
            "methods": {},
            "summary": {
                "total_execution_time": total_time,
                "methods_compared": len(results),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        best_score = -float('inf')
        best_method = None
        
        for method, result in results.items():
            if result is None:
                comparison["methods"][method] = {
                    "success": False,
                    "error": "Optimization failed for this method"
                }
                continue
                
            comparison["methods"][method] = convert_result_to_response(
                result, method
            ).dict()
            
            if result.fitness_score > best_score:
                best_score = result.fitness_score
                best_method = method
        
        comparison["summary"]["best_method"] = best_method
        comparison["summary"]["best_score"] = best_score if best_method else None
        
        logger.info(f"Comparison completed, best: {best_method} ({best_score:.4f})")
        
        return JSONResponse(content=comparison)
        
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Comparison failed", "message": str(e)}
        )


@app.post("/generate-synthetic")
async def generate_synthetic_data(request: SyntheticDataRequest):
    """
    Generate synthetic test data using EnhancedMetroDataGenerator.
    
    Useful for testing the optimization API without providing real data.
    """
    try:
        logger.info(f"Generating synthetic data for {request.num_trainsets} trainsets")
        
        # Generate data
        generator = EnhancedMetroDataGenerator(num_trainsets=request.num_trainsets)
        data = generator.generate_complete_enhanced_dataset()
        
        # Remove trainset_profiles as it contains non-serializable datetime objects
        # and is not needed for optimization
        data_for_response = {
            "trainset_status": data["trainset_status"],
            "fitness_certificates": data["fitness_certificates"],
            "job_cards": data["job_cards"],
            "component_health": data["component_health"],
            "metadata": data.get("metadata", {})
        }
        
        logger.info(f"Generated synthetic data with {len(data['trainset_status'])} trainsets")
        
        return JSONResponse(content={
            "success": True,
            "data": data_for_response,
            "metadata": {
                "num_trainsets": len(data['trainset_status']),
                "num_fitness_certificates": len(data['fitness_certificates']),
                "num_job_cards": len(data['job_cards']),
                "num_component_health": len(data['component_health']),
                "generated_at": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Synthetic data generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Data generation failed", "message": str(e)}
        )


@app.post("/validate")
async def validate_data(request: ScheduleOptimizationRequest):
    """
    Validate input data structure without running optimization.
    
    Returns validation results and suggestions for fixing issues.
    """
    try:
        # Convert to dict
        data = convert_pydantic_to_dict(request)
        
        # Validate
        validation_errors = DataValidator.validate_data(data)
        
        if not validation_errors:
            return {
                "valid": True,
                "message": "Data structure is valid",
                "num_trainsets": len(request.trainset_status),
                "num_certificates": len(request.fitness_certificates),
                "num_job_cards": len(request.job_cards),
                "num_component_health": len(request.component_health)
            }
        
        return {
            "valid": False,
            "validation_errors": validation_errors,
            "suggestions": [
                "Check that all trainset_ids are consistent across sections",
                "Ensure operational_status values are valid (Available, In-Service, Maintenance, Standby, Out-of-Order)",
                "Verify certificate expiry dates are in ISO format",
                "Confirm component wear_level is between 0-100"
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Validation failed", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
