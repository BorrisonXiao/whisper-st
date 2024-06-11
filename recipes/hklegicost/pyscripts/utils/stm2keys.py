#!/usr/bin/env python
import os
import logging
from tqdm import tqdm
import local.data_prep.stm as stm
import argparse
from pathlib import Path


def _stm2keys(stm_file: Path, scp_file: Path):
    with open(stm_file, 'r') as f:
        lines = f.readlines()

    with open(scp_file, 'w') as f:
        for line in lines:
            uttid = stm.parse_StmUtterance(line).utterance_id(stereo=True)
            print(f"{uttid}", file=f)


def stm2keys(input_dir: Path, output_dir: Path, split: str):
    # Retrive all .stm files that match the split name under the input directory
    stm_files = [input_dir / _path for _path in os.listdir(input_dir) if _path.endswith(".stm") and split in _path]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for stm_file in tqdm(stm_files, desc="Converting stm to scp"):
        # Generate the scp file name
        scp_file = output_dir / "keys.scp"
        _stm2keys(Path(stm_file), scp_file)


def main():
    parser = argparse.ArgumentParser(description='Convert stm to scp')
    parser.add_argument('-i', '--input-dir', type=Path,
                        required=True, help='Input stm file directory.')
    parser.add_argument('-o', '--output-dir', type=Path,
                        required=True, help='Output scp file directory.')
    parser.add_argument('-s', '--split', type=str,
                        default='dev', help='Split name.')
    args = parser.parse_args()

    stm2keys(input_dir=args.input_dir,
            output_dir=args.output_dir, split=args.split)


if __name__ == '__main__':
    main()
