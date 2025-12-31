OUTPUT_DIR := /Users/poldrack/data_unsynced/BCBS/simple_workflow/wf_snakemake
DOCKER_IMAGE := poldrack/simple-workflow
DOCKER_TAG := latest

.PHONY: run run-conda run-apptainer report graph clean docker-build docker-push

clean:
	-rm -rf .snakemake
	-rm -rf $(OUTPUT_DIR)/*

run:
	uv run snakemake --cores 1 -d $(OUTPUT_DIR)

run-conda:
	uv run snakemake --cores 1 --sdm conda -d $(OUTPUT_DIR)

run-apptainer:
	uv run snakemake --cores 1 --sdm apptainer -d $(OUTPUT_DIR)

lint:
	snakemake --lint --cores 1 -d $(OUTPUT_DIR)

dryrun:
	snakemake --dry-run --cores 1 -d $(OUTPUT_DIR)

dryrun-apptainer:
	snakemake --dry-run --cores 1 --sdm apptainer -d $(OUTPUT_DIR)

report:
	snakemake --report $(OUTPUT_DIR)/report.html -d $(OUTPUT_DIR)

graph:
	snakemake --rulegraph -d $(OUTPUT_DIR) --cores 2 | dot -Tpng -Gdpi=300 > $(OUTPUT_DIR)/rulegraph.png

export-env:
	-mkdir envs
	conda env export -n bettercode > envs/bettercode.yml

# Container image management
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-push:
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
