#!/usr/bin/env python

from datasets import Dataset, Features, Audio, Value
import argparse
from pathlib import Path
from collections import defaultdict
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil


def csv2utts(csv_file, dumpdir, fs, srcdir, make_cuts=True, min_duration=0.5, max_duration=999999, max_fname_len=60):
    dset = Path(csv_file).stem
    audio_dir = srcdir / "audio"
    tmpdir = dumpdir / dset / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [l.strip().split('\t') for l in lines[1:]]

    res = defaultdict(list)
    cached_audio = (None, None)
    for line in tqdm(lines, desc=f"Creating raw segments in {tmpdir}..."):
        audio_fname, spkid, start, end, transcript, translation = line

        # Filter out utterances that are too short or too long
        dur = float(end) - float(start)
        if dur < min_duration or dur > max_duration:
            continue

        _fname = Path(audio_fname).stem
        # Trim the filename if it is too long
        if len(_fname) > max_fname_len:
            start_pos = len(_fname) - max_fname_len
            _fname = _fname[start_pos:]
        uttid = f"{spkid}__{_fname}__{int(float(start)*100):06d}-{int(float(end)*100):06d}"

        tmp_audio = tmpdir / f"{uttid}.wav"
        if make_cuts:
            # Cut the original audio file into utterances
            # based on the start and end time
            # Save the utterances in the tmpdir
            # The utterance file name is the uttid
            # The utterance file format is wav
            if audio_fname != cached_audio[0]:
                cached_audio = (audio_fname, librosa.load(
                    audio_dir / audio_fname, sr=fs))

            # Make the cut
            sf.write(tmp_audio, cached_audio[1][0][int(
                float(start)*fs):int(float(end)*fs)], fs)

        # Save the uttid, spkid, start, end, transcript, translation
        res["audio"].append(str(tmp_audio))
        res["spkid"].append(spkid)
        res["transcript"].append(transcript)
        res["translation"].append(translation)
        res["uttid"].append(uttid)
        res["src_lang"].append("yue")
        res["tgt_lang"].append("en")

    return res


def create_hf_dset(dset, srcdir, dumpdir, min_duration, max_duration, fs=16000, rm_tmp=False, make_cuts=True, max_fname_len=60):
    print(f"Creating {dset} dataset")
    # Step 1: Parse the csv file to get the
    # audio file name, speaker id, start time, end time, transcript and translation
    # separated by the '\t' character
    # Also the original recording will be cut into utterances
    # based on the start and end time
    csv_file = srcdir / "splits" / f"{dset}.csv"
    utts = csv2utts(csv_file, srcdir=srcdir, dumpdir=dumpdir,
                    min_duration=min_duration, max_duration=max_duration,
                    fs=fs, make_cuts=make_cuts, max_fname_len=max_fname_len)

    # Step 2: Create the dataset
    features = Features({
        "audio": Audio(
            sampling_rate=fs,
        ),
        "spkid": Value(dtype="string"),
        "uttid": Value(dtype="string"),
        "transcript": Value(dtype="string"),
        "translation": Value(dtype="string"),
        "src_lang": Value(dtype="string"),
        "tgt_lang": Value(dtype="string"),
    })

    # Store the uttids in a plain text file for parallel processing
    uttids = utts["uttid"]
    with open(dumpdir / dset / "uttids", "w") as f:
        print("\n".join(uttids), file=f)

    hf_dset = Dataset.from_dict(utts, features=features)
    hf_dset.save_to_disk(dumpdir / dset)

    if rm_tmp:
        shutil.rmtree(dumpdir / dset / "tmp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset', type=str, choices=['train', 'test', 'dev-asr-0',
                        'dev-asr-1', 'dev-asr-2', 'dev-mt-0', 'dev-mt-1', 'dev-mt-2'], default="test")
    parser.add_argument('--srcdir', type=Path,
                        default="/home/cxiao7/research/legicost/export",
                        help="The directory where the csv and audio files are stored")
    parser.add_argument('--dumpdir', type=Path, default="dump")
    parser.add_argument('--min-duration', type=float, default=0.0)
    parser.add_argument('--max-duration', type=float, default=999999)
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--rm-tmp', action='store_true')
    parser.add_argument('--make-cuts', action='store_true')
    parser.add_argument('--max-fname-len', type=int, default=60)
    args = parser.parse_args()

    create_hf_dset(dset=args.dset, srcdir=args.srcdir, dumpdir=args.dumpdir,
                   min_duration=args.min_duration, max_duration=args.max_duration,
                   fs=args.fs, rm_tmp=args.rm_tmp, make_cuts=args.make_cuts, max_fname_len=args.max_fname_len)


if __name__ == '__main__':
    main()
