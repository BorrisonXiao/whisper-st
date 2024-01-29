#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from local.data_prep.stm import StmUtterance
from tqdm import tqdm


class MergedUtterance:
    def __init__(self, uttid, duration, wav_paths, transcript, translation):
        self.uttid = uttid
        self.duration = duration
        self.wav_paths = wav_paths
        self.transcript = transcript
        self.translation = translation

    def __repr__(self):
        return f"Uttid: {self.uttid}, duration: {self.duration}, wav_paths: {self.wav_paths}, transcript: {self.transcript}, translation: {self.translation}"


def parse_keyfile(keyfile):
    with open(keyfile, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # A new utterance in the keyfile is defined by 4 lines:
    # The first line is in the format of:
    # uttid duration
    # The second line is a space-delimited list of paths to the wav files
    # The third line is the merged transcript
    # The fourth line is the merged translation
    utts = []
    for i in range(0, len(lines), 4):
        uttid, duration = lines[i].strip().split()
        wav_paths = lines[i + 1].strip().split()
        transcript = lines[i + 2].strip()
        translation = lines[i + 3].strip()
        utts.append(MergedUtterance(uttid, duration,
                    wav_paths, transcript, translation))
    return utts


def join_wavs(wav_paths, output_path):
    """
    Join the wavs in wav_paths into a single wav file.
    """
    wavs = []
    for wav_path in wav_paths:
        wav, sr = librosa.load(wav_path, sr=None)
        wavs.append(wav)
    wav = np.concatenate(wavs)
    sf.write(output_path, wav, sr)


def generate(keyfile, dumpdir, output_dir, dump_prefix=None):
    # Step 1: Parse the keyfile
    utts = parse_keyfile(keyfile)

    # Step 2: Concate the wavs for each utterance and create the stm file
    datadir = dumpdir / "data"
    Path(datadir).mkdir(parents=True, exist_ok=True)
    sr_stm_path = dumpdir / "merged.sr.stm"
    st_stm_path = dumpdir / "merged.st.stm"
    with open(sr_stm_path, "w", encoding="utf-8") as f_sr, open(st_stm_path, "w", encoding="utf-8") as f_st:
        for utt in tqdm(utts):
            abs_wav_path = (datadir / f"{utt.uttid}.wav").absolute()
            join_wavs(utt.wav_paths, abs_wav_path)
            sr_utt = StmUtterance(
                filename=abs_wav_path,
                channel="A",  # A placeholder as we don't have the channel information
                # A placeholder as we don't have the speaker information
                speaker=f"{utt.uttid.split('-')[0]}-{utt.uttid.split('-')[1]}",
                start_time=0,
                stop_time=float(utt.duration),
                transcript=utt.transcript,
            )
            if dump_prefix:
                splitted = str(sr_utt.filename).split("/dump/")
                sr_utt.filename = Path(dump_prefix) / splitted[1]
            st_utt = StmUtterance(
                filename=abs_wav_path,
                channel="A",  # A placeholder as we don't have the channel information
                # A placeholder as we don't have the speaker information
                speaker=f"{utt.uttid.split('-')[0]}-{utt.uttid.split('-')[1]}",
                start_time=0,
                stop_time=float(utt.duration),
                transcript=utt.translation,
            )
            if dump_prefix:
                splitted = str(st_utt.filename).split("/dump/")
                st_utt.filename = Path(dump_prefix) / splitted[1]
            print(sr_utt, file=f_sr)
            print(st_utt, file=f_st)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=Path, help="Key file")
    parser.add_argument("--dumpdir", type=Path, help="Dump directory")
    parser.add_argument("--dump-prefix", type=str, default="/exp/cxiao/scale23/dump_scale23",
                        help="Dump directory used to replace the stm path prefix for permission issues")
    parser.add_argument("--output_dir", type=Path, help="Output directory")
    args = parser.parse_args()
    generate(keyfile=args.keyfile, dumpdir=args.dumpdir,
             dump_prefix=args.dump_prefix, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
