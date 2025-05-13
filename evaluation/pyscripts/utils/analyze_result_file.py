#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import csv

def _get_parser():
    """
    pyscripts/utils/analyze_result_file.py \
        --ref ${score_dir}/ref.tc.rm \
        --hyp ${score_dir}/hyp.tc.rm \
        --scores ${score_dir}/result.lc.rm.txt \
        --output ${score_dir}/analysis.lc.rm.csv
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ref', type=Path, required=True, help='Reference file')
    parser.add_argument('--hyp', type=Path, required=True, help='Hypothesis file')
    parser.add_argument('--scores', type=Path, required=True, help='Scores file')
    parser.add_argument('--output', type=Path, required=True, help='Output file')

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    
    # Reads the ref bitext file
    with open(args.ref, "r") as f:
        ref_lines = f.readlines()
    
    # Reads the hyp bitext file
    with open(args.hyp, "r") as f:
        hyp_lines = f.readlines()
        
    # Reads the scores file
    with open(args.scores, "r") as f:
        scores = f.readlines()
        
    results = []
    for i, (ref_line, hyp_line, score_line) in enumerate(zip(ref_lines, hyp_lines, scores)):
        ref = ref_line.strip()
        hyp = hyp_line.strip()
        score = score_line.strip()
        
        results.append({
            "ref": ref,
            "hyp": hyp,
            "score": score
        })
    
    # Writes the results to a csv file
    with open(args.output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["score", "ref", "hyp"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    main()
