"""
FastAPI Service for Metro Train Schedule Generation
Provides endpoints for synthetic data generation and schedule optimization
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from datetime import datetime
import logging

from ..core.models import (
    DaySchedule, ScheduleRequest, Route, TrainHealthStatus
)
from ..generators.metro_generator import MetroDataGenerator
from ..optimizers.schedule_optimizer import MetroScheduleOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Metro Train Scheduling API",
    description="Generate synthetic metro data and optimize daily train schedules",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Metro Train Scheduling API",
        "version": "1.0.0",
        "endpoints": {
            "schedule": "/api/v1/schedule",
            "generate": "/api/v1/generate",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "metro-scheduling-api"
    }


@app.post("/api/v1/generate", response_model=DaySchedule)
async def generate_schedule(request: ScheduleRequest):
    """
    Generate optimized daily train schedule
    
    Args:
        request: Schedule request with date, train count, and optimization parameters
        
    Returns:
        DaySchedule: Complete optimized schedule with all trainset assignments
        
    Example:
        POST /api/v1/generate
        {
            "date": "2025-10-25",
            "num_trains": 30,
            "num_stations": 25,
            "min_service_trains": 22,
            "min_standby_trains": 3
        }
    """
    try:
        logger.info(f"Generating schedule for {request.date} with {request.num_trains} trains")
        
        # Initialize data generator
        generator = MetroDataGenerator(
            num_trains=request.num_trains,
            num_stations=request.num_stations
        )
        
        # Generate route
        route = generator.generate_route(request.route_name)
        logger.info(f"Generated route: {route.name} with {len(route.stations)} stations")
        
        # Generate or use provided train health data
        if request.train_health_overrides:
            train_health = request.train_health_overrides
        else:
            train_health = generator.generate_train_health_statuses()
        
        logger.info(f"Train health data: {len(train_health)} trains initialized")
        
        # Initialize optimizer
        optimizer = MetroScheduleOptimizer(
            date=request.date,
            num_trains=request.num_trains,
            route=route,
            train_health=train_health,
            depot_name=request.depot_name
        )
        
        # Optimize schedule
        schedule = optimizer.optimize_schedule(
            min_service_trains=request.min_service_trains,
            min_standby=request.min_standby_trains,
            max_daily_km=request.max_daily_km_per_train
        )
        
        logger.info(
            f"Schedule generated: {schedule.schedule_id}, "
            f"{schedule.fleet_summary.revenue_service} trains in service, "
            f"{schedule.optimization_metrics.total_planned_km} km planned"
        )
        
        return schedule
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating schedule: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/generate/quick")
async def generate_quick_schedule(
    date: str = "2025-10-25",
    num_trains: int = 25,
    num_stations: int = 25
):
    """
    Quick schedule generation with default parameters
    
    Query Parameters:
        - date: Schedule date (YYYY-MM-DD)
        - num_trains: Number of trains in fleet (default: 25)
        - num_stations: Number of stations on route (default: 25)
    """
    request = ScheduleRequest(
        date=date,
        num_trains=num_trains,
        num_stations=num_stations
    )
    return await generate_schedule(request)


@app.get("/api/v1/route/{num_stations}")
async def get_route_info(num_stations: int = 25):
    """
    Get metro route information
    
    Args:
        num_stations: Number of stations to include (default: 25)
        
    Returns:
        Route information with all stations
    """
    try:
        generator = MetroDataGenerator(num_stations=num_stations)
        route = generator.generate_route()
        
        return {
            "route": route.model_dump(),
            "one_way_time_minutes": int((route.total_distance_km / route.avg_speed_kmh) * 60),
            "round_trip_time_minutes": int((route.total_distance_km / route.avg_speed_kmh) * 60 * 2) + route.turnaround_time_minutes * 2
        }
    except Exception as e:
        logger.error(f"Error generating route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/trains/health/{num_trains}")
async def get_train_health(num_trains: int = 25):
    """
    Generate train health status data
    
    Args:
        num_trains: Number of trains (default: 25)
        
    Returns:
        List of train health statuses
    """
    try:
        generator = MetroDataGenerator(num_trains=num_trains)
        health_data = generator.generate_train_health_statuses()
        
        summary = {
            "total": len(health_data),
            "fully_healthy": sum(1 for h in health_data if h.is_fully_healthy),
            "partial": sum(1 for h in health_data if not h.is_fully_healthy and h.available_hours),
            "unavailable": sum(1 for h in health_data if not h.is_fully_healthy and not h.available_hours)
        }
        
        return {
            "summary": summary,
            "trains": [h.dict() for h in health_data]
        }
    except Exception as e:
        logger.error(f"Error generating train health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/depot/layout")
async def get_depot_layout():
    """Get depot bay layout information"""
    try:
        generator = MetroDataGenerator()
        layout = generator.generate_depot_layout()
        
        return {
            "depot": "Muttom_Depot",
            "layout": layout,
            "total_bays": sum(len(bays) for bays in layout.values())
        }
    except Exception as e:
        logger.error(f"Error generating depot layout: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/schedule/example")
async def get_example_schedule():
    """Get an example schedule for demonstration"""
    request = ScheduleRequest(
        date=datetime.now().strftime("%Y-%m-%d"),
        num_trains=30,
        num_stations=25,
        min_service_trains=22,
        min_standby_trains=4
    )
    return await generate_schedule(request)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
