#!/usr/bin/env python3
"""
CLI tool to convert Parquet files to JSON format.

This script provides a command-line interface to read Parquet files and convert
them to JSON format with various output options.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow is required. Install with: pip install pyarrow")
    sys.exit(1)

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_dataframe(
    input_source: str,
    is_hf_dataset: bool = False,
    hf_split: str = "train",
    hf_config: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from either a Parquet file or Hugging Face dataset.
    
    Args:
        input_source: Path to Parquet file or Hugging Face dataset name
        is_hf_dataset: Whether to load from Hugging Face
        hf_split: Split to load from HF dataset (default: "train")
        hf_config: Configuration name for HF dataset (optional)
    
    Returns:
        pandas DataFrame
    """
    if is_hf_dataset:
        if not HF_AVAILABLE:
            print("Error: datasets library is required for Hugging Face datasets.", file=sys.stderr)
            print("Install with: pip install datasets", file=sys.stderr)
            sys.exit(1)
        
        print(f"Loading Hugging Face dataset: {input_source}", file=sys.stderr)
        if hf_config:
            print(f"  Config: {hf_config}", file=sys.stderr)
        print(f"  Split: {hf_split}", file=sys.stderr)
        
        try:
            dataset = load_dataset(input_source, name=hf_config, split=hf_split)
            df = dataset.to_pandas()
            print(f"Loaded dataset with {len(df)} rows", file=sys.stderr)
            return df
        except Exception as e:
            print(f"Error loading Hugging Face dataset: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Check if input file exists
        if not os.path.exists(input_source):
            print(f"Error: Input file '{input_source}' does not exist.", file=sys.stderr)
            sys.exit(1)
        
        # Read the Parquet file
        print(f"Reading Parquet file: {input_source}", file=sys.stderr)
        return pd.read_parquet(input_source)


def parse_parquet_to_json(
    input_file: str,
    output_file: Optional[str] = None,
    pretty: bool = False,
    records_format: bool = False,
    max_rows: Optional[int] = None,
    is_hf_dataset: bool = False,
    hf_split: str = "train",
    hf_config: Optional[str] = None
) -> None:
    """
    Convert a Parquet file or Hugging Face dataset to JSON format.
    
    Args:
        input_file: Path to the input Parquet file or HF dataset name
        output_file: Path to the output JSON file (if None, prints to stdout)
        pretty: Whether to format JSON with indentation
        records_format: Whether to output as records format (list of objects)
        max_rows: Maximum number of rows to process (None for all)
        is_hf_dataset: Whether to load from Hugging Face
        hf_split: Split to load from HF dataset
        hf_config: Configuration name for HF dataset
    """
    try:
        # Load the data
        df = load_dataframe(input_file, is_hf_dataset, hf_split, hf_config)
        
        # Filter rows where score column is 1
        if 'score' in df.columns:
            original_count = len(df)
            
            # Check if score column contains arrays/lists or scalar values
            sample_score = df['score'].iloc[0] if len(df) > 0 else None
            if isinstance(sample_score, (list, tuple)) or (hasattr(sample_score, '__len__') and not isinstance(sample_score, str)):
                # Score is an array - filter rows where ALL elements are 1
                df = df[df['score'].apply(lambda x: all(val == 1 for val in x) if hasattr(x, '__iter__') and not isinstance(x, str) else x == 1)]
                print(f"Filtered to {len(df)} rows where all scores are 1 (from {original_count} total rows)", file=sys.stderr)
            else:
                # Score is a scalar value
                df = df[df['score'] == 1]
                print(f"Filtered to {len(df)} rows where score=1 (from {original_count} total rows)", file=sys.stderr)
        else:
            print("Warning: 'score' column not found in the data. No filtering applied.", file=sys.stderr)
        
        # Limit rows if specified
        if max_rows is not None:
            df = df.head(max_rows)
            print(f"Limited to first {max_rows} rows", file=sys.stderr)
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns", file=sys.stderr)
        
        # Convert to JSON - use pandas to_json for better serialization
        if records_format:
            json_output = df.to_json(orient='records', date_format='iso')
        else:
            json_output = df.to_json(orient='records', date_format='iso')
        
        # If pretty formatting is requested, parse and reformat
        if pretty:
            json_obj = json.loads(json_output)
            json_output = json.dumps(json_obj, indent=2, ensure_ascii=False)
        
        # Write output
        if output_file:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"JSON output written to: {output_file}", file=sys.stderr)
        else:
            print(json_output)
            
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


def get_parquet_info(input_file: str) -> None:
    """
    Display information about a Parquet file.
    
    Args:
        input_file: Path to the Parquet file
    """
    try:
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        # Read Parquet metadata
        parquet_file = pq.ParquetFile(input_file)
        df_sample = pd.read_parquet(input_file, nrows=5)  # Read first 5 rows for preview
        
        print(f"Parquet file: {input_file}")
        print(f"Number of rows: {parquet_file.metadata.num_rows}")
        print(f"Number of columns: {parquet_file.metadata.num_columns}")
        print(f"File size: {os.path.getsize(input_file):,} bytes")
        print("\nColumns:")
        for i, col in enumerate(parquet_file.schema):
            print(f"  {i+1}. {col.name} ({col.physical_type})")
        
        print("\nFirst 5 rows preview:")
        print(df_sample.to_string(index=False))
        
    except Exception as e:
        print(f"Error reading file info: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Parquet files or Hugging Face datasets to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert parquet to JSON (output to stdout)
  python parse-OPC.py input.parquet

  # Convert to JSON file with pretty formatting
  python parse-OPC.py input.parquet -o output.json --pretty

  # Convert first 1000 rows only
  python parse-OPC.py input.parquet -o output.json --max-rows 1000

  # Show file information
  python parse-OPC.py input.parquet --info

  # Load from Hugging Face dataset
  python parse-OPC.py "username/dataset-name" --hf -o output.json --pretty

  # Load specific split and config from HF dataset
  python parse-OPC.py "username/dataset-name" --hf --hf-split validation --hf-config default -o output.json

  # Convert all parquet files in a directory
  for file in *.parquet; do
    python parse-OPC.py "$file" -o "${file%.parquet}.json" --pretty
  done
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input Parquet file or Hugging Face dataset name (e.g., 'username/dataset-name')"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (if not specified, prints to stdout)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Format JSON with indentation for readability"
    )
    
    parser.add_argument(
        "--records",
        action="store_true",
        help="Output as records format (array of objects)"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Maximum number of rows to process"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about the Parquet file instead of converting"
    )
    
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Load from Hugging Face dataset instead of Parquet file"
    )
    
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split to load from Hugging Face dataset (default: train)"
    )
    
    parser.add_argument(
        "--hf-config",
        help="Configuration name for Hugging Face dataset (optional)"
    )
    
    args = parser.parse_args()
    
    if args.info:
        if args.hf:
            print("Error: --info flag is not supported with Hugging Face datasets", file=sys.stderr)
            sys.exit(1)
        get_parquet_info(args.input_file)
    else:
        parse_parquet_to_json(
            input_file=args.input_file,
            output_file=args.output,
            pretty=args.pretty,
            records_format=args.records,
            max_rows=args.max_rows,
            is_hf_dataset=args.hf,
            hf_split=args.hf_split,
            hf_config=args.hf_config
        )


if __name__ == "__main__":
    main()