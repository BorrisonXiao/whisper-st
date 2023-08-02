#!/usr/bin/env python

import argparse
from pathlib import Path
from local.stm import parse_StmUtterance
import os


def stm2txt(input_dir, out_dir):
    langdir = input_dir
    for peft_method in os.listdir(langdir):
        _peftdir = langdir / peft_method
        for train_set in os.listdir(_peftdir):
            train_set_dir = _peftdir / train_set
            for setting in os.listdir(train_set_dir):
                if not str(setting).endswith("_merged"):
                    continue
                setting_dir = train_set_dir / setting
                for dset in os.listdir(setting_dir):
                    dset_dir = setting_dir / dset
                    stm_file = dset_dir / f"hyp_mt.stm"
                    with open(stm_file, "r") as f:
                        lines = f.readlines()
                    _outdir = out_dir / peft_method / train_set / setting / dset
                    _outdir.mkdir(parents=True, exist_ok=True)
                    with open(_outdir / "hyp_mt.txt", "w") as f:
                        # The lines will be sorted based on the start time and rootfile name
                        lines = sorted(lines, key=lambda x: int(Path(parse_StmUtterance(x).filename).stem.split("-")[-1].split("_")[1]))
                        lines = sorted(lines, key=lambda x: Path(parse_StmUtterance(x).filename).stem.split("-")[1])
                        if "fleurs" in dset:
                            breakpoint()
                        for line in lines:
                            utt = parse_StmUtterance(line)
                            f.write(utt.transcript + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    stm2txt(input_dir=args.input_dir, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
