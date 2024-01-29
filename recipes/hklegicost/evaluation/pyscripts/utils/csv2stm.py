#!/usr/bin/env python

import argparse
from pathlib import Path

_data_base_dir = Path("/home/cxiao7/research/legicost/export")


def parse_csv(csv_file, audio_dir, task="asr"):
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip().split('\t') for l in lines[1:]]

    res = {}
    for line in lines:
        audio_fname, spkid, start, end, transcript, translation = line
        _fname = Path(audio_fname).stem
        uttid = f"{spkid}__{_fname}__{int(float(start)*100):06d}-{int(float(end)*100):06d}"
        text = transcript if task == "asr" else translation

        res[uttid] = {"audio": audio_dir / audio_fname, "spkid": spkid, "start": start,"end": end, "text": text}
    
    return res


def csv2stm(csv_file, stm_file, data_base_dir, stereo=False, task="asr"):
    """
    Convert the csv file to stm file.
    """
    # Parse the csv file
    csv_dict = parse_csv(csv_file, data_base_dir / "audio", task=task)
    # Write the stm file
    with open(stm_file, "w", encoding="utf-8") as f:
        for uttid, utt_dict in csv_dict.items():
            if stereo:
                print(f"{utt_dict['audio']} 1 {utt_dict['spkid']} {utt_dict['start']} {utt_dict['end']} <O> {utt_dict['text']}", file=f)
            else:
                print(f"{utt_dict['audio']} {utt_dict['spkid']} {utt_dict['start']} {utt_dict['end']} <O> {utt_dict['text']}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description='Convert a csv file to a stm file')
    parser.add_argument("-i", "--input", type=Path, help="Input csv file", required=True)
    parser.add_argument(
        "-o", "--output", type=Path, help="Output stm file", required=True)
    parser.add_argument(
        "-d", "--data-base-dir", type=Path, help="Data base directory", default=_data_base_dir)
    parser.add_argument(
        "--stereo", help="Stereo stm", action="store_true")
    parser.add_argument(
        "--task", help="Task", choices=["asr", "mt"], default="asr")

    args = parser.parse_args()
    csv2stm(csv_file=args.input, stm_file=args.output, data_base_dir=args.data_base_dir, stereo=args.stereo, task=args.task)


if __name__ == "__main__":
    main()
