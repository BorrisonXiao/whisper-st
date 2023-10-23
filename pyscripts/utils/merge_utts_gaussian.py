#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch

SPLITS = ["dev", "dev1", "dev2", "train-all_sp",
          "train-cts_sp", "train-ood_sp"]

torch.random.manual_seed(1234)


class Utterance:
    def __init__(self, uttid, spkid, start, end, root_filename, filepath, channel, transcript=None, translation=None):
        self.uttid = uttid
        self.start = start
        self.end = end
        self.root_filename = root_filename
        self.filepath = filepath
        self.spkid = spkid
        self.dur = end - start
        self.transcript = transcript
        self.translation = translation
        self.channel = channel

    def __repr__(self):
        return f"Uttid: {self.uttid}, spkid: {self.spkid}, start: {self.start}, end: {self.end}, root_filename: {self.root_filename}, dur: {self.dur}, filepath: {self.filepath}, channel: {self.channel}, transcript: {self.transcript}, translation: {self.translation}"

    def set_transcript(self, transcript):
        self.transcript = transcript

    def set_translation(self, translation):
        self.translation = translation


def parse_utt(line):
    """
    Parse the utterance line from the scp file.
    """
    uttid, filepath = line.strip().split(maxsplit=1)
    splitted = uttid.split("-")
    prefix = None
    if splitted[0].startswith("sp") and splitted[0][2] != "_":
        prefix = splitted[0]
        splitted = splitted[1:]
    if len(splitted) == 2:
        root_fname, info = splitted
        # For some OOD training data, the spkid is not present or encoded in a different way
        spkid = root_fname
    elif len(splitted) == 3:
        spkid, root_fname, info = splitted
    elif len(splitted) == 4:
        spkid1, spkid2, root_fname, info = splitted
        spkid = f"{spkid1}-{spkid2}"
    elif len(splitted) >= 5 and len(splitted[-2]) == 3 and splitted[-2].isnumeric() and len(splitted[-3]) == 3 and splitted[-3].isnumeric():
        # Spanish OOD training data (europarl)
        info = splitted[-1]
        root_fname = f"{splitted[-4]}-{splitted[-3]}-{splitted[-2]}"
        spkid = "-".join(splitted[:-4])
    else:
        raise ValueError("Invalid line: {}".format(line))
    if prefix is not None:
        root_fname = f"{prefix}-{root_fname}"
    channel, start, end = info.split("_")
    start = int(start) / 100
    end = int(end) / 100
    return Utterance(uttid, spkid, start, end, root_fname, filepath, channel)


def merge_utts(src_lang, tgt_lang, input_base_dir, output_base_dir, splits=SPLITS, num_outputs=1, mean=15, std=7, t_min=5, t_max=25):
    """
    Merge the utterances so that the durations of the utterances are Gaussian distributed, which seems to be helpful
    based on Amir's previous study. This is also to avoid the problem of having too many short or long utterances (labels),
    as the label length is doubled for the Bayesian decomposed training.
    """
    print('Merging utts for {}...'.format(src_lang))
    for split in splits:
        asr_scp = input_base_dir / f"{src_lang}.{split}.wav.scp"
        asr_text = input_base_dir / f"{src_lang}.{split}.text"
        st_text = input_base_dir / f"{src_lang}-{tgt_lang}.{split}.text"
        # If the file does not exist, skip it
        if not asr_scp.exists():
            print(f"Warning: {asr_scp} does not exist. Skipping...")
            continue
        print(f"Processing {asr_scp}...")
        # Read and parse the scp files
        with open(asr_scp, "r") as f:
            asr_utts = [parse_utt(line.strip()) for line in f.readlines()]
        # Convert to dict for easy access
        asr_utts_dict = {utt.uttid: utt for utt in asr_utts}
        with open(asr_text, "r") as f:
            for line in f:
                uttid, text = line.strip().split(" ", maxsplit=1)
                asr_utts_dict[uttid].set_transcript(text)
        with open(st_text, "r") as f:
            for line in f:
                uttid, text = line.strip().split(" ", maxsplit=1)
                asr_utts_dict[uttid].set_translation(text)
        # Sort the utterances by start time and root filename
        asr_utts_dict = dict(
            sorted(asr_utts_dict.items(), key=lambda x: x[1].start))
        asr_utts_dict = dict(sorted(asr_utts_dict.items(),
                             key=lambda x: x[1].root_filename))

        # Merge the utterances
        merged_utts = {}
        cur_dur = 0
        to_merge = []
        cur_root_fname = None
        for i, (uttid, utt) in enumerate(asr_utts_dict.items()):
            # Generate the target length based on the Gaussian distribution
            t = min(max(torch.normal(mean, std, size=(1,)).item(), t_min), t_max)
            if cur_dur + utt.dur > t and to_merge != [] or cur_root_fname is not None and cur_root_fname != utt.root_filename:
                # Merge the utterances in to_merge, note that the spkid is no longer faithful as it is the spkid of the last utt
                new_uttid = f"{utt.spkid}-{utt.root_filename}-{utt.channel}_{round(to_merge[0].start * 100):08d}_{round(to_merge[-1].end * 100):08d}"
                merged_utts[new_uttid] = (to_merge, cur_dur)
                # Reset cur_dur and to_merge
                cur_dur = 0
                to_merge = []

            # Add the current utt to to_merge
            to_merge.append(utt)
            cur_dur += utt.dur
            cur_root_fname = utt.root_filename

            # Check if we have reached the end of the dict, if so, write the last merged utt
            if i == len(asr_utts_dict) - 1:
                new_uttid = f"{utt.spkid}-{utt.root_filename}-{utt.channel}_{round(to_merge[0].start * 100):08d}_{round(to_merge[-1].end * 100):08d}"
                merged_utts[new_uttid] = (to_merge, cur_dur)

        # Write the merged utts to file
        _logdir = output_base_dir / split
        Path(_logdir).mkdir(parents=True, exist_ok=True)
        # Split the merged utts into num_outputs subsets
        subsets = [{} for _ in range(num_outputs)]
        for i in range(num_outputs):
            for j, (uttid, utts) in enumerate(merged_utts.items()):
                if j % num_outputs == i:
                    subsets[i][uttid] = utts

        for i, subset in enumerate(subsets):
            with open(_logdir / f"keys.{split}.{i + 1}.scp", "w") as f:
                for uttid, utts in subset.items():
                    print(uttid, utts[1], file=f)
                    print(" ".join([utt.filepath for utt in utts[0]]), file=f)
                    print(
                        " ".join([utt.transcript for utt in utts[0]]), file=f)
                    print(
                        " ".join([utt.translation for utt in utts[0]]), file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', type=str, default='ara')
    parser.add_argument('--tgt_lang', type=str, default='eng')
    parser.add_argument('--input_base_dir', type=Path, default='data')
    parser.add_argument('--output_base_dir', type=Path, default='data')
    parser.add_argument('--num_outputs', type=int, default=1,
                        help='Number of outputs to generate for distributed processing')
    parser.add_argument('--splits', type=str, nargs='+', default=SPLITS)
    parser.add_argument('--mean', type=int, default=15)
    parser.add_argument('--std', type=int, default=7)
    parser.add_argument('--t_min', type=int, default=5)
    parser.add_argument('--t_max', type=int, default=25)
    args = parser.parse_args()

    merge_utts(src_lang=args.src_lang,
               tgt_lang=args.tgt_lang,
               input_base_dir=args.input_base_dir,
               output_base_dir=args.output_base_dir,
               num_outputs=args.num_outputs,
               splits=args.splits,
               mean=args.mean,
               std=args.std,
               t_min=args.t_min,
               t_max=args.t_max)


if __name__ == '__main__':
    main()
