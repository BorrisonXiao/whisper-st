#!/usr/bin/env python
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
from tqdm import tqdm
import shutil


def export_wav(dumpdir, outdir):
    """
    Helper script for exporting the wav_raw.scp files (from relative to absolute path).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for langdir in dumpdir.iterdir():
        lang = langdir.name
        rawdir = langdir / "raw"
        # Iterate over testsets first
        for testsetdir in rawdir.glob("*_test"):
            setname = testsetdir.name
            srcfile = testsetdir / "wav_raw.scp"
            if not srcfile.exists():
                print("Warning: {} does not exist.".format(srcfile))
                continue
            with open(srcfile, "r") as f:
                lines = f.readlines()
            with open(outdir / f"{lang}.{setname}.wav.scp", "w") as f:
                for line in lines:
                    utt, wav = line.strip().split(maxsplit=1)

                    # Remove the leading "dump/" in the path
                    wav = re.sub(r"^dump/", "", wav)
                    print(f"{utt} {dumpdir / wav}", file=f)
            # Copy over the text file
            shutil.copy(testsetdir / "text", outdir / f"{lang}.{setname}.text")
            shutil.copy(testsetdir / "text.tc.eng", outdir / f"{lang}-eng.{setname}.text")
        # Iterate over org/(dev|train) sets
        for dsetdir in tqdm(rawdir.glob("org/*")):
            setname = dsetdir.name
            srcfile = dsetdir / "wav_raw.scp"
            if not srcfile.exists():
                print("Warning: {} does not exist.".format(srcfile))
                continue
            with open(srcfile, "r") as f:
                lines = f.readlines()
            with open(outdir / f"{lang}.{setname}.wav.scp", "w") as f:
                for line in lines:
                    utt, wav = line.strip().split(maxsplit=1)

                    # Remove the leading "dump/" in the path
                    wav = re.sub(r"^dump/", "", wav)
                    print(f"{utt} {dumpdir / wav}", file=f)
            # Copy over the text file
            shutil.copy(dsetdir / "text", outdir / f"{lang}.{setname}.text")
            shutil.copy(dsetdir / "text.tc.eng", outdir / f"{lang}-eng.{setname}.text")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumpdir", help="dump directory",
                        type=Path, default="/exp/cxiao/scale23/dump_scale23")
    parser.add_argument("--outdir", help="output directory",
                        type=Path, default="dump/export")
    args = parser.parse_args()

    export_wav(dumpdir=args.dumpdir, outdir=args.outdir)


if __name__ == "__main__":
    main()
