BENCHMARK_DIR := benchmarks
OUTPUT_DIR := benchmark_output
PYTHON := python3

.PHONY: benchmarks fleet-benchmark optimizer-benchmark service-benchmark constraint-benchmark clean-output help quick-benchmark

# Create output directory and run all benchmarks
benchmarks: fleet-benchmark optimizer-benchmark service-benchmark
	@echo "=============================================="
	@echo "All benchmarks complete!"
	@echo "Results saved to: $(OUTPUT_DIR)/"
	@echo "=============================================="

# Fleet Utilization Benchmark
fleet-benchmark:
	@echo "=============================================="
	@echo "Running Fleet Utilization Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/fleet_utilization && $(PYTHON) benchmark_fleet_utilization.py
	@echo "Fleet benchmark complete. Output in $(OUTPUT_DIR)/"

# Optimizer Performance Benchmark
optimizer-benchmark:
	@echo "=============================================="
	@echo "Running Optimizer Performance Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/optimizer_performance && $(PYTHON) benchmark_optimizers.py
	@echo "Optimizer benchmark complete. Output in $(OUTPUT_DIR)/"

# Service Quality Benchmark
service-benchmark:
	@echo "=============================================="
	@echo "Running Service Quality Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/service_quality && $(PYTHON) benchmark_service_quality.py
	@echo "Service quality benchmark complete. Output in $(OUTPUT_DIR)/"

# Constraint Satisfaction Benchmark (requires schedule data, run separately)
constraint-benchmark:
	@echo "=============================================="
	@echo "Running Constraint Satisfaction Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/constraint_satisfaction && $(PYTHON) benchmark_constraints.py
	@echo "Constraint benchmark complete. Output in $(OUTPUT_DIR)/"

# Quick benchmark (fast tests with fewer configurations)
quick-benchmark:
	@echo "=============================================="
	@echo "Running Quick Optimizer Benchmark..."
	@echo "=============================================="
	cd $(BENCHMARK_DIR)/optimizer_performance && $(PYTHON) benchmark_optimizers.py --quick
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
	@echo "  benchmarks          - Run all benchmarks (default)"
	@echo "  fleet-benchmark     - Run fleet utilization benchmark"
	@echo "  optimizer-benchmark - Run optimizer performance benchmark"
	@echo "  service-benchmark   - Run service quality benchmark"
	@echo "  constraint-benchmark - Run constraint satisfaction benchmark"
	@echo "  quick-benchmark     - Run quick optimizer benchmark"
	@echo "  clean-output        - Clean benchmark output files"
	@echo "  clean               - Remove output directory completely"
	@echo "  help                - Show this help message"
	@echo ""
	@echo "All outputs are saved to: $(OUTPUT_DIR)/"
