#!/usr/bin/env python

import argparse
from pathlib import Path
from local.stm import parse_StmUtterance

data_base_dir = Path("/exp/scale23/data//audio/ara/iwslt22/")


def text2stm(text_file, stm_file, ref_stm_file=None, dset="dev", merge_utt=False):
    """
    Note that this works only for the Tunisian data.
    """
    def uttid2stm(uttid):
        # The uttid can be parsed as follows:
        # <spkid>-<filename>-<channel>_<start>_<end>
        # Example: 997612-20171122_161929_20929_B-A_00063778_00063887
        # The channel is either A or B
        splitted = uttid.split("-")
        if len(splitted) == 3:
            spkid, filename, info = splitted
        elif len(splitted) == 4:
            if merge_utt:
                spkid, filename, _info, info = splitted
            elif dset == "fleurs":
                spkid1, spkid2, filename, info = splitted
                spkid = f"{spkid1}-{spkid2}"
            else:
                raise ValueError("Invalid uttid: {}".format(uttid))
        elif len(splitted) == 5 and dset == "fleurs" and merge_utt:
            spkid1, spkid2, filename, _info, info = splitted
            spkid = f"{spkid1}-{spkid2}"
        else:
            raise ValueError("Invalid uttid: {}".format(uttid))
        channel, start, end = info.split("_")
        start = int(start) // 100
        end = int(end) // 100
        return spkid, filename, channel, start, end

    def stm2uttid(stm):
        # Reverse construction of uttid from stm
        stm_utt = parse_StmUtterance(stm)
        return stm_utt.utterance_id(stereo=True)

    # Step 1: Read the text file
    text_dict = {}
    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        splitted = line.strip().split(" ", maxsplit=1)
        if len(splitted) == 1:
            uttid = splitted[0]
            text = " "
        else:
            uttid, text = splitted
        text_dict[uttid] = {"text": text, "stm": uttid2stm(uttid)}

    # Step 2: Read the reference stm file if provided
    if ref_stm_file is not None:
        ref_stm_dict = {}
        with open(ref_stm_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            uttid = stm2uttid(line.strip())
            ref_stm_dict[uttid] = " ".join(line.strip().split()[:6])

    # Step 3: Generate the stm file
    with open(stm_file, "w", encoding="utf-8") as f:
        if ref_stm_dict is not None:
            for uttid, header in ref_stm_dict.items():
                text = text_dict[uttid]["text"]
                f.write(f"{header} {text}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert kaldi text file to stm files')
    parser.add_argument("-i", "--input", help="Input text file", required=True)
    parser.add_argument(
        "-o", "--output", help="Output stm file", required=True)
    parser.add_argument(
        "-r", "--ref", help="Reference stm file, based on which the output is sorted", default=None)
    parser.add_argument(
        "-d", "--dset", help="Dataset name", default="dev")
    parser.add_argument(
        "--merge-utt", help="Merge utterances", action="store_true")

    args = parser.parse_args()
    text2stm(text_file=args.input, stm_file=args.output,
             ref_stm_file=args.ref, dset=args.dset, merge_utt=args.merge_utt)


if __name__ == "__main__":
    main()
