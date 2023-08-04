#!/usr/bin/env python

import argparse
from pathlib import Path
from local.stm import parse_StmUtterance
import os
from bs4 import BeautifulSoup
import re
import copy


def xml2stm(input_dir, out_dir, refdir, mode="st"):
    model = input_dir.stem
    # Get language from the path <path>/<lang>/aligned/large-v2
    lang = input_dir.parent.parent.stem
    for peft_method in os.listdir(input_dir):
        _peftdir = input_dir / peft_method
        for train_set in os.listdir(_peftdir):
            train_set_dir = _peftdir / train_set
            for dset in os.listdir(train_set_dir):
                dset_dir = train_set_dir / dset
                stm_file = dset_dir / f"aligned.xml"
                with open(stm_file, "r") as f:
                    data = f.read()
                    xml_data = BeautifulSoup(data, "xml")
                tgt_lang = "eng" if mode == "st" else lang
                _refdir = refdir if "_test" not in dset else refdir / "testsets" / "cts"
                ref_stm = _refdir / f"{mode}.{lang}-{tgt_lang}.{dset}.stm" if "_test" not in dset else _refdir / f"{mode}.{lang}-{tgt_lang}.{dset.replace('_test', '.test')}.stm"
                with open(ref_stm, "r") as f:
                    ref_lines = f.readlines()
                # The reference stm is used to re-sort the lines
                ref_utts = [parse_StmUtterance(line.strip()) for line in ref_lines]
                
                _segs = xml_data.find_all("seg")
                texts = {seg['id']: re.sub('<.*?>', '', seg.text).strip() for seg in _segs}
                _outdir = out_dir / model / peft_method / train_set / dset
                _outdir.mkdir(parents=True, exist_ok=True)
                with open(_outdir / f"hyp_{mode}.stm", "w") as f:
                    for utt in ref_utts:
                        uttid = utt.utterance_id(stereo=True)
                        _utt = copy.deepcopy(utt)
                        _utt.transcript = texts[uttid]
                        f.write(str(_utt) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", "-m", type=str, default="st", choices=["st", "sr"])
    parser.add_argument("--refdir", "-r", type=Path, required=True)
    args = parser.parse_args()

    xml2stm(input_dir=args.input_dir, out_dir=args.output_dir, mode=args.mode, refdir=args.refdir)


if __name__ == "__main__":
    main()
