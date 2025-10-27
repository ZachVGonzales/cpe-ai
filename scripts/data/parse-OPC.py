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


def parse_parquet_to_json(
    input_file: str,
    output_file: Optional[str] = None,
    pretty: bool = False,
    records_format: bool = False,
    max_rows: Optional[int] = None
) -> None:
    """
    Convert a Parquet file to JSON format.
    
    Args:
        input_file: Path to the input Parquet file
        output_file: Path to the output JSON file (if None, prints to stdout)
        pretty: Whether to format JSON with indentation
        records_format: Whether to output as records format (list of objects)
        max_rows: Maximum number of rows to process (None for all)
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
            sys.exit(1)
        
        # Read the Parquet file
        print(f"Reading Parquet file: {input_file}", file=sys.stderr)
        df = pd.read_parquet(input_file)
        
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
        description="Convert Parquet files to JSON format",
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

  # Convert all parquet files in a directory
  for file in *.parquet; do
    python parse-OPC.py "$file" -o "${file%.parquet}.json" --pretty
  done
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input Parquet file"
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
    
    args = parser.parse_args()
    
    if args.info:
        get_parquet_info(args.input_file)
    else:
        parse_parquet_to_json(
            input_file=args.input_file,
            output_file=args.output,
            pretty=args.pretty,
            records_format=args.records,
            max_rows=args.max_rows
        )


if __name__ == "__main__":
    main()