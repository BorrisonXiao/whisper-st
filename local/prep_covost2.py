#!/usr/bin/env python
import os
from datasets import load_dataset
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

AUDIO_SAMPLING_RATE = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--save-dir", type=Path, required=True,
                        help="Path to the output location.")
    parser.add_argument("--lang", type=str, default="fr",
                        help="Language code for the dataset.")
    args = parser.parse_args()
    
    lang = args.lang
    
    # Load the dataset
    dataset = load_dataset("covost2", f"{lang}_en", data_dir=args.data_dir)
    print(f"Dataset loaded: {dataset}")
    
    files = defaultdict(dict)
    spkid = defaultdict(dict)
    duration = defaultdict(dict)
    dur_strs = defaultdict(lambda: "00000000")
    transcripts = defaultdict(dict)
    translations = defaultdict(dict)
    # for split in ["test", "validation", "train"]:
    for split in ["test"]:
        valid_indices = set()
        for i in tqdm(range(len(dataset[split]))):
            try:
                utt = dataset[split][i]
                valid_indices.add(i)

                dur = len(utt["audio"]["array"]) / AUDIO_SAMPLING_RATE
                dur_str = f"{int(dur * 100):08d}"
                uttid = utt["client_id"] + "_" + Path(utt["file"]).stem + f"_00000000_{int(dur * 100):08d}"
                files[split][uttid] = utt["file"]
                spkid[split][uttid] = utt["client_id"]
                # Compute the duration based on the sample rate and audio array length
                duration[split][uttid] = dur
                dur_strs[Path(utt["file"]).stem] = dur_str
                transcripts[split][uttid] = utt["sentence"].strip('\"')
                translations[split][uttid] = utt["translation"]
            except Exception as e:
                print(f"Error processing {split} split at index {i}: {e}")
        # Select only the valid indices
        dataset[split] = dataset[split].select(list(valid_indices))
        # Print the number of valid samples
        print(f"Number of valid samples in {split} split: {len(valid_indices)}")

    # Split the dataset and rename the columns
    # e.g. sentence -> transcript
    # Also insert new columns: src_lang, tgt_lang, utternace_id. Note that the utternace_id is the client_id plus the stem of the audio file plus the <start-time>_<end-time>
    # Delete the columns: client_id, file, sentence
    # dataset = dataset.map(lambda x: {"transcript": x["sentence"].strip('\"'), "src_lang": f"{lang}", "tgt_lang": "en", "uttid": x["client_id"] + "_" + Path(x["file"]).stem + f"_00000000_{dur_strs[Path(x['file']).stem]}"}, remove_columns=["client_id", "file", "sentence"])
    
    # # Save each split to disk
    # dataset["validation"].save_to_disk(args.save_dir / f"{lang}.validation")
    # dataset["test"].save_to_disk(args.save_dir / f"{lang}.test")
    # dataset["train"].save_to_disk(args.save_dir / f"{lang}.train")
    
    # Store a wav.scp like file, kaldi-style text file, and a stm file for the dev and test split
    # for split in ["test", "validation", "train"]:
    for split in ["test"]:
        with open(args.save_dir / f"{split}.wav.scp", "w") as f:
            for uttid, file in files[split].items():
                print(f"{uttid} {file}", file=f)
        with open(args.save_dir / f"{split}.text", "w") as f:
            for uttid, transcript in transcripts[split].items():
                print(f"{uttid} {transcript}", file=f)
        # BUG: Here the correct logic is to derive the .text file from the .stm file using the .utterance_id function
        with open(args.save_dir / f"{split}.src.stm", "w") as f:
            for uttid, transcript in transcripts[split].items():
                dur = duration[split][uttid]
                print(f"{files[split][uttid]} A {spkid[split][uttid]} 0.00 {int(dur * 100) / 100:.2f} <O> {transcript}", file=f)
        with open(args.save_dir / f"{split}.tgt.stm", "w") as f:
            for uttid, translation in translations[split].items():
                dur = duration[split][uttid]
                print(f"{files[split][uttid]} A {spkid[split][uttid]} 0.00 {int(dur * 100) / 100:.2f} <O> {translation}", file=f)


if __name__ == "__main__":
    main()
