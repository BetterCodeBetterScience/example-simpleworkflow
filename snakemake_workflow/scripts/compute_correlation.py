"""Snakemake script for computing correlation matrix."""

import sys
from pathlib import Path

import pandas as pd

# Add workflow directory to path for local simple_workflow module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simple_workflow import compute_correlation_matrix

def main():
    """Compute Spearman correlation matrix."""
    # ruff: noqa: F821
    input_path = Path(snakemake.input[0]).expanduser()
    output_path = Path(snakemake.output[0]).expanduser()
    method = snakemake.params.method

    # Load data
    df = pd.read_csv(input_path, index_col=0)
    print(f"Loaded {df.shape} from {input_path}")

    # Compute correlation
    corr_matrix = compute_correlation_matrix(df, method=method)
    print(f"Computed {method} correlation matrix: {corr_matrix.shape}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
