#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv
from matplotlib import pyplot as plt

def _get_parser():
    """
    pyscripts/utils/compare_bleu.py \
        --analysis1 /home/hltcoe/cxiao/st/evaluation/scores_ft/mtl/st/hf_whisper_large-v2/spa/lora/train-cts_sp/merged_org/fisher_test/analysis.lc.rm.csv \
        --analysis2 /home/hltcoe/cxiao/st/evaluation/scores_ft/mml/st/hf_whisper_large-v2/spa/lora_0.8_0.2/train-cts_sp/merged_org/fisher_test_asr_prompt/analysis.lc.rm.csv \
        --system1 mtl \
        --system2 cts \
        --output-dir /home/hltcoe/cxiao/scale23/whisper/recipe/st/evaluation/analysis
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--analysis1', type=Path, help='Path to the first analysis file')
    parser.add_argument('--analysis2', type=Path, help='Path to the second analysis file')
    parser.add_argument('--system1', type=str, help='Name of the first system')
    parser.add_argument('--system2', type=str, help='Name of the second system')
    parser.add_argument('--threshold', type=float, default=0, help='Threshold to determine the difference')
    parser.add_argument('--output-dir', type=Path, help='Path to the output directory')

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    
    # Reads the first analysis csv file in the format of score, reference, hypothesis
    with open(args.analysis1, 'r') as f:
        reader = csv.reader(f)
        analysis1 = list(reader)
        f1name = args.system1
    
    # Reads the second analysis csv file in the format of score, reference, hypothesis
    with open(args.analysis2, 'r') as f:
        reader = csv.reader(f)
        analysis2 = list(reader)
        f2name = args.system2
        
    # For the results, split into 3 subsets, one that analysis1 is better than analysis2, and the other way around, and the tie
    f1_better = []
    f2_better = []
    tie = []
    for i in range(len(analysis1)):
        # Skip the header
        if i == 0:
            continue
        # Store the difference in the score up to 2 decimal places
        score_diff = float(analysis1[i][0]) - float(analysis2[i][0])
        score_diff_str = f'{score_diff:.2f}'
        if score_diff > args.threshold:
            f1_better.append([score_diff_str, analysis1[i][1], analysis1[i][2], analysis2[i][2]])
        elif score_diff < -args.threshold:
            f2_better.append([score_diff_str, analysis1[i][1], analysis1[i][2], analysis2[i][2]])
        else:
            tie.append([analysis1[i][0], analysis1[i][1], analysis1[i][2], analysis2[i][2]])
            
    # Plot each subset's as histogram w.r.t. the length of the reference
    # Save the all the histograms as a single png file (subplots)
    # plt.figure(figsize=(20, 10))
    # plt.subplot(3, 1, 1)
    # plt.hist([len(x[1].split()) for x in f1_better], bins=20, alpha=0.5, label=f'{f1name} better')
    # plt.hist([len(x[1].split()) for x in f2_better], bins=20, alpha=0.5, label=f'{f2name} better')
    # plt.hist([len(x[1].split()) for x in tie], bins=20, alpha=0.5, label='tie')
    # plt.legend(loc='upper right')
    # plt.title('Histogram of the length of the reference')
    # plt.xlabel('Length of the reference')
    # plt.ylabel('Frequency')
    # # Save the histogram as a png file
    # plt.savefig(args.output_dir / 'histogram.png')
    output_dir = args.output_dir / f'{f1name}_vs_{f2name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    axes[0].hist([len(x[1].split()) for x in f1_better], bins=20, alpha=0.5, label=f'{f1name} better')
    axes[1].hist([len(x[1].split()) for x in f2_better], bins=20, alpha=0.5, label=f'{f2name} better')
    axes[2].hist([len(x[1].split()) for x in tie], bins=20, alpha=0.5, label='tie')
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'Histogram of the length of the reference (# = {len(f1_better)})')
    axes[0].set_xlabel('Length of the reference')
    axes[0].set_ylabel('Frequency')
    axes[1].legend(loc='upper right')
    axes[1].set_title(f'Histogram of the length of the reference (# = {len(f2_better)})')
    axes[1].set_xlabel('Length of the reference')
    axes[1].set_ylabel('Frequency')
    axes[2].legend(loc='upper right')
    axes[2].set_title(f'Histogram of the length of the reference (# = {len(tie)})')
    axes[2].set_xlabel('Length of the reference')
    # Set the same y-axis for all subplots based on the largest y-axis
    max_y = 500
    axes[0].set_ylim(0, max_y)
    axes[1].set_ylim(0, max_y)
    axes[2].set_ylim(0, max_y)
    plt.tight_layout()
    plt.savefig(output_dir / 'histogram.png')
            
    # Creates a new csv file to store the comparison results
    f1_better_csv = output_dir / f'{f1name}_better.csv'
    with open(f1_better_csv, 'w') as f:
        writer = csv.writer(f)
        # Columns: score difference, reference, hypothesis1, hypothesis2
        # Write headers first
        writer.writerow(['score difference', 'reference', f'{f1name} hypothesis', f'{f2name} hypothesis'])
        writer.writerows(f1_better)
    
    f2_better_csv = output_dir / f'{f2name}_better.csv'
    with open(f2_better_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['score difference', 'reference', f'{f1name} hypothesis', f'{f2name} hypothesis'])
        writer.writerows(f2_better)
        
    tie_csv = output_dir / 'tie.csv'
    with open(tie_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['score difference', 'reference', f'{f1name} hypothesis', f'{f2name} hypothesis'])
        writer.writerows(tie)


if __name__ == "__main__":
    main()
