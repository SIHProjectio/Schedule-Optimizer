# Metro Train Scheduling System - Benchmark Makefile
# All benchmark outputs go to benchmark_output/ directory

BENCHMARK_DIR := benchmarks
OUTPUT_DIR := benchmark_output
PYTHON := python3

# Timestamp for unique output filenames
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

.PHONY: all benchmarks fleet-benchmark optimizer-benchmark service-benchmark constraint-benchmark clean-output help

# Create output directory and run all benchmarks
benchmarks: $(OUTPUT_DIR) fleet-benchmark optimizer-benchmark service-benchmark
	@echo "=============================================="
	@echo "All benchmarks complete!"
	@echo "Results saved to: $(OUTPUT_DIR)/"
	@echo "=============================================="

# Create output directory
$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)
	@echo "Created output directory: $(OUTPUT_DIR)/"

# Fleet Utilization Benchmark
fleet-benchmark: $(OUTPUT_DIR)
	@echo "=============================================="
	@echo "Running Fleet Utilization Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/fleet_utilization && \
		$(PYTHON) benchmark_fleet_utilization.py && \
		mv -f fleet_utilization_benchmark_*.json ../../$(OUTPUT_DIR)/ 2>/dev/null || true && \
		mv -f fleet_utilization_report_*.txt ../../$(OUTPUT_DIR)/ 2>/dev/null || true
	@echo "Fleet benchmark complete. Output in $(OUTPUT_DIR)/"

# Optimizer Performance Benchmark
optimizer-benchmark: $(OUTPUT_DIR)
	@echo "=============================================="
	@echo "Running Optimizer Performance Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/optimizer_performance && \
		$(PYTHON) benchmark_optimizers.py && \
		mv -f optimizer_benchmark_*.json ../../$(OUTPUT_DIR)/ 2>/dev/null || true && \
		mv -f optimizer_performance_report_*.txt ../../$(OUTPUT_DIR)/ 2>/dev/null || true
	@echo "Optimizer benchmark complete. Output in $(OUTPUT_DIR)/"

# Service Quality Benchmark
service-benchmark: $(OUTPUT_DIR)
	@echo "=============================================="
	@echo "Running Service Quality Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/service_quality && \
		$(PYTHON) benchmark_service_quality.py && \
		mv -f service_quality_benchmark_*.json ../../$(OUTPUT_DIR)/ 2>/dev/null || true
	@echo "Service quality benchmark complete. Output in $(OUTPUT_DIR)/"

# Constraint Satisfaction Benchmark (requires schedule data, run separately)
constraint-benchmark: $(OUTPUT_DIR)
	@echo "=============================================="
	@echo "Running Constraint Satisfaction Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/constraint_satisfaction && \
		$(PYTHON) benchmark_constraints.py && \
		mv -f constraint_satisfaction_results*.json ../../$(OUTPUT_DIR)/ 2>/dev/null || true
	@echo "Constraint benchmark complete. Output in $(OUTPUT_DIR)/"

# Quick benchmark (fast tests with fewer configurations)
quick-benchmark: $(OUTPUT_DIR)
	@echo "=============================================="
	@echo "Running Quick Optimizer Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/optimizer_performance && \
		$(PYTHON) benchmark_optimizers.py --quick && \
		mv -f optimizer_benchmark_*.json ../../$(OUTPUT_DIR)/ 2>/dev/null || true && \
		mv -f optimizer_performance_report_*.txt ../../$(OUTPUT_DIR)/ 2>/dev/null || true
	@echo "Quick benchmark complete. Output in $(OUTPUT_DIR)/"

# Clean output directory
clean-output:
	@echo "Cleaning benchmark outputs..."
	rm -rf $(OUTPUT_DIR)/*
	@echo "Output directory cleaned."

# Full clean including output directory
clean: clean-output
	rm -rf $(OUTPUT_DIR)
	@echo "Removed $(OUTPUT_DIR)/ directory."

# Help
help:
	@echo "Metro Scheduling System Benchmarks"
	@echo "==================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all               - Run all benchmarks (default)"
	@echo "  benchmarks        - Run all benchmarks"
	@echo "  fleet-benchmark   - Run fleet utilization benchmark"
	@echo "  optimizer-benchmark - Run optimizer performance benchmark"
	@echo "  service-benchmark - Run service quality benchmark"
	@echo "  constraint-benchmark - Run constraint satisfaction benchmark"
	@echo "  quick-benchmark   - Run quick optimizer benchmark"
	@echo "  clean-output      - Clean benchmark output files"
	@echo "  clean             - Remove output directory completely"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "All outputs are saved to: $(OUTPUT_DIR)/"
