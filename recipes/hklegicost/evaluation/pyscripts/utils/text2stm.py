#!/usr/bin/env python

import argparse
from pathlib import Path
from local.stm import parse_StmUtterance
import warnings

data_base_dir = Path("/exp/scale23/data//audio/ara/iwslt22/")


def _truncate_uttid(uttid, max_fname_len=60):
    # Truncate the uttid if it is too long
    # The uttid can be parsed as follows:
    # <spkid>__<filename>__<start>-<end>
    # Example: 保安局副局長__保安局局長_M16110051_4_00-33-32_00-43-27__000095-001584
    # The channel is either 1 or None
    splitted = uttid.split("__")
    if len(splitted) == 3:
        spkid, filename, info = splitted
    else:
        raise ValueError("Invalid uttid: {}".format(uttid))
    start, end = info.split("-")
    start = int(start) / 100
    end = int(end) / 100
    _fname = Path(filename).stem
    # Trim the filename if it is too long
    if len(_fname) > max_fname_len:
        start_pos = len(_fname) - max_fname_len
        _fname = _fname[start_pos:]
    return f"{spkid}__{_fname}__{int(start*100):06d}-{int(end*100):06d}"


def text2stm(text_file, stm_file, ref_stm_file=None, dset="dev", merge_utt=False):
    """
    Note that this works only for the Tunisian data.
    """
    def uttid2stm(uttid):
        # The uttid can be parsed as follows:
        # <spkid>__<filename>__<start>-<end>
        # Example: 保安局副局長__保安局局長_M16110051_4_00-33-32_00-43-27__000095-001584
        # The channel is either 1 or None
        splitted = uttid.split("__")
        if len(splitted) == 3:
            spkid, filename, info = splitted
        else:
            raise ValueError("Invalid uttid: {}".format(uttid))
        start, end = info.split("-")
        start = int(start) / 100
        end = int(end) / 100
        return spkid, filename, "A", start, end

    def stm2uttid(stm):
        # Reverse construction of uttid from stm
        stm_utt = parse_StmUtterance(stm)
        return stm_utt.utterance_id(stereo=False)

    # Step 1: Read the hypothesis text file
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
    missing_utt_count = 0
    with open(stm_file, "w", encoding="utf-8") as f:
        if ref_stm_dict is not None:
            for uttid, header in ref_stm_dict.items():
                uttid = _truncate_uttid(uttid)
                if uttid not in text_dict:
                    warnings.warn(f"{uttid} is not in the hypothesis file...")
                    missing_utt_count += 1
                    text = " "
                else:
                    text = text_dict[uttid]["text"]
                f.write(f"{header} {text}\n")
    print(f"{missing_utt_count} utterances are missing in the hypothesis file...")


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
