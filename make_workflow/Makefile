# Simple Correlation Workflow using Make
#
# Usage:
#   make all              - Run full workflow
#   make clean            - Remove output directory

OUTPUT_DIR ?= ./output

.PHONY: all clean

all: download_data filter_data join_data compute_correlation generate_heatmap

download_data:
	mkdir -p $(OUTPUT_DIR)/data $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures
	uv run python scripts/download_data.py $(OUTPUT_DIR)/data

filter_data:
	uv run python scripts/filter_data.py $(OUTPUT_DIR)/data

join_data:
	uv run python scripts/join_data.py $(OUTPUT_DIR)/data

compute_correlation:
	uv run python scripts/compute_correlation.py $(OUTPUT_DIR)/data $(OUTPUT_DIR)/results

generate_heatmap:
	uv run python scripts/generate_heatmap.py $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures

$(OUTPUT_DIR)/correlation_heatmap.png: generate_heatmap
	
clean:
	rm -rf $(OUTPUT_DIR)
