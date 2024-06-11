#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Cihan Xiao (2024 Johns Hopkins University)

"""
Convert the Bemba main data to stm format.
"""

from pathlib import Path
import argparse
from local.data_prep.stm import StmUtterance
import csv
from tqdm import tqdm


def create_stm(srcdir, outdir, dset, mode, src_lang, tgt_lang):
    _mode = "asr" if mode == "sr" else "st"
    _target_lang = tgt_lang if mode == "st" else src_lang
    _outdir= outdir / _mode
    _outdir.mkdir(parents=True, exist_ok=True)
    
    csv_file = srcdir / "splits" / f"{dset}.csv"
    audio_base = srcdir / "audio"
    utts = []
    print(f"Processing {csv_file}...")
    with open(csv_file, "r") as f:
        _f = csv.reader(f, delimiter='\t')
        for i, line in enumerate(_f):
            if i == 0:
                continue
            # Remove all the \n from the transcript and translation
            _audio_id = Path(line[0]).stem
            _spkid = line[1]
            _start_time = float(line[2])
            _end_time = float(line[3])
            _transcript = line[4].replace("\n", "")
            _translation = line[5].replace("\n", "")
            utts.append(dict(audio_id=_audio_id, spkid=_spkid, start_time=_start_time, end_time=_end_time, transcript=_transcript.strip(), translation=_translation.strip()))
    
    print(f"Processing {dset} split...")
    with open(_outdir / f"{mode}.{src_lang}-{_target_lang}.{dset}.stm", "w") as f:
        for utt in tqdm(utts):
            audio_file = audio_base / f"{utt['audio_id']}.wav"
            start_time = utt["start_time"]
            end_time = utt["end_time"]
            duration = end_time - start_time
            spkid = utt["spkid"]
            channel = "A"
            stm_utt = StmUtterance(
                filename=audio_file,
                channel=channel,
                speaker=spkid,
                start_time=start_time,
                stop_time=end_time,
                transcript=utt["transcript"] if mode == "sr" else utt["translation"],
            )
            print(stm_utt, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--srcdir", type=Path, required=True, help="The source directory of the main data.")
    parser.add_argument("-o", "--outdir", type=Path, required=True, help="The output directory of the stm file.")
    parser.add_argument("--mode", type=str, default="sr", help="The mode of the data.")
    parser.add_argument("--dset", type=str, default="train", help="The supervision to process.")
    parser.add_argument("--src-lang", type=str, default="cmn", help="The source language.")
    parser.add_argument("--tgt-lang", type=str, default="eng", help="The target language.")
    args = parser.parse_args()

    create_stm(srcdir=args.srcdir, outdir=args.outdir, dset=args.dset, mode=args.mode, src_lang=args.src_lang, tgt_lang=args.tgt_lang)


if __name__ == "__main__":
    main()