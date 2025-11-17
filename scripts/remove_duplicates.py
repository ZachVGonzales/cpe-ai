#!/usr/bin/env python3
"""
Script to remove duplicate problem IDs from a JSON file.
Keeps the first occurrence of each problem_id and removes subsequent duplicates.
"""

import json
import argparse
from pathlib import Path


def remove_duplicates(input_file, output_file):
    """
    Remove duplicate problem IDs from JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Read the JSON file
    print(f"Reading from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries before deduplication: {len(data)}")
    
    # Track seen problem IDs and keep only first occurrence
    seen_ids = set()
    unique_data = []
    duplicates_found = []
    
    for entry in data:
        problem_id = entry.get('problem_id')
        
        if problem_id not in seen_ids:
            seen_ids.add(problem_id)
            unique_data.append(entry)
        else:
            duplicates_found.append(problem_id)
    
    print(f"Total entries after deduplication: {len(unique_data)}")
    print(f"Duplicates removed: {len(duplicates_found)}")
    
    if duplicates_found:
        print(f"\nDuplicate problem IDs found:")
        # Count occurrences of each duplicate
        from collections import Counter
        duplicate_counts = Counter(duplicates_found)
        for prob_id, count in sorted(duplicate_counts.items()):
            print(f"  - {prob_id}: {count} duplicate(s)")
    
    # Write to output file
    print(f"\nWriting to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Remove duplicate problem IDs from a JSON file'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSON file'
    )
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to output JSON file (default: input_file with "_deduplicated" suffix)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    # Generate output filename if not provided
    if args.output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_deduplicated{input_path.suffix}"
    else:
        output_path = Path(args.output_file)
    
    remove_duplicates(input_path, output_path)
    
    return 0


if __name__ == '__main__':
    exit(main())
