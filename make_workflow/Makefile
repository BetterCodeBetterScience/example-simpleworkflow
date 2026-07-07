# Simple Correlation Workflow using Make
#
# Usage:
#   make all              - Run full workflow
#   make clean            - Remove output directory

OUTPUT_DIR ?= ./output

.PHONY: all clean

all: $(OUTPUT_DIR)/figures/correlation_heatmap.png

$(OUTPUT_DIR)/data/demographics.csv $(OUTPUT_DIR)/data/meaningful_variables.csv:
	@echo "Downloading data..."
	mkdir -p $(OUTPUT_DIR)/data $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures
	uv run python scripts/download_data.py $(OUTPUT_DIR)/data

$(OUTPUT_DIR)/data/demographics_numerical.csv: $(OUTPUT_DIR)/data/demographics.csv
	@echo "Filtering demographics data..."
	uv run python scripts/filter_data.py $(OUTPUT_DIR)/data

$(OUTPUT_DIR)/data/meaningful_variables_numerical.csv: $(OUTPUT_DIR)/data/meaningful_variables.csv
	@echo "Filtering meaningful variables data..."
	uv run python scripts/filter_data.py $(OUTPUT_DIR)/data

$(OUTPUT_DIR)/data/joined_data.csv: $(OUTPUT_DIR)/data/demographics_numerical.csv $(OUTPUT_DIR)/data/meaningful_variables_numerical.csv
	@echo "Joining data..."
	uv run python scripts/join_data.py $(OUTPUT_DIR)/data

$(OUTPUT_DIR)/results/correlation_matrix.csv: $(OUTPUT_DIR)/data/joined_data.csv
	@echo "Computing correlation..."
	uv run python scripts/compute_correlation.py $(OUTPUT_DIR)/data $(OUTPUT_DIR)/results

$(OUTPUT_DIR)/figures/correlation_heatmap.png: $(OUTPUT_DIR)/results/correlation_matrix.csv
	@echo "Generating heatmap..."
	uv run python scripts/generate_heatmap.py $(OUTPUT_DIR)/results $(OUTPUT_DIR)/figures
	
clean:
	rm -rf $(OUTPUT_DIR)
