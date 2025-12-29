# Simple Correlation Workflow using Make
#
# Usage:
#   make all              - Run full workflow
#   make clean            - Remove output directory

OUTPUT_DIR ?= ./output

.PHONY: all clean

all:
	mkdir -p $(OUTPUT_DIR)/data $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures
	uv run python scripts/download_data.py $(OUTPUT_DIR)/data
	uv run python scripts/filter_data.py $(OUTPUT_DIR)/data
	uv run python scripts/join_data.py $(OUTPUT_DIR)/data
	uv run python scripts/compute_correlation.py $(OUTPUT_DIR)/data $(OUTPUT_DIR)/results
	uv run python scripts/generate_heatmap.py $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures

clean:
	rm -rf $(OUTPUT_DIR)
