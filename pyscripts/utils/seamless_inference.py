#!/usr/bin/env python3
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import AutoProcessor, SeamlessM4TForSpeechToText, SeamlessM4TForTextToText
import torchaudio
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset
import traceback
import json
import warnings

LANGS = {
    "eng": "eng",
    "spa": "spa",
    "cmn": "cmn",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
}

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def inference(
    keyfile,
    src_lang,
    tgt_lang,
    output_dir,
    model_name,
    task,
    dset=None,
    batch_size=1,
    num_beams=2,
    rank=0,
):
    cuda_id = rank
    device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
    print(
        f"Decoding with batch_size={batch_size}, num_beams={num_beams} on {device}")

    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        f"facebook/hf-seamless-m4t-{model_name}")
    if task == "MT":
        model = SeamlessM4TForTextToText.from_pretrained(
            f"facebook/hf-seamless-m4t-{model_name}").to(device)
    else:
        model = SeamlessM4TForSpeechToText.from_pretrained(
            f"facebook/hf-seamless-m4t-{model_name}").to(device)

    print(f"model.device: {model.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"batch_size: {batch_size}")
    print(f"num_beams: {num_beams}")
    
    # Load the HF dataset
    ds = load_from_disk(dset)
    last_uttid = ds[-1]["uttid"]

    # Load the keyfile
    if keyfile is not None:
        with open(keyfile, "r") as f:
            lines = f.readlines()
        _keys = [line.strip().split(maxsplit=1)[0] for line in lines]
        last_uttid = _keys[-1]
        keys = set(_keys)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "text"
    
    tgt_lang = src_lang if task == "ASR" else tgt_lang
    print(
        f"Decoding {task} with src_lang={src_lang}, tgt_lang={tgt_lang} to {output}")

    with open(output, "w") as f:
        total_len = len(ds) if keyfile is None else len(keys)
        pbar = tqdm(range(total_len))
        batch = []
        uttids = []
        for utt in ds:
            uttid = utt["uttid"]
            if keyfile is not None and uttid not in keys:
                continue
            # Accumulate the batch
            uttids.append(uttid)
            batch.append(utt)
            if len(batch) < batch_size and uttid != last_uttid:
                continue
            # Process the batch
            if task != "MT":
                audios = [utt["audio"]["array"] for utt in batch]
                input_features = processor(
                    audios=audios,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    src_lang=src_lang,
                    sample_rate=16000,
                    ).to(device)
            else:
                texts = [utt["transcript"] for utt in batch]
                input_features = processor(
                    text=texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    src_lang=src_lang,
                    ).to(device)
            # Generate token ids
            predicted_ids = model.generate(**input_features,tgt_lang=tgt_lang,num_beams=num_beams,)
            # Decode token ids to text
            hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            for i, hyp in enumerate(hyps):
                print(uttids[i], hyp, file=f)
            f.flush()
            pbar.update(len(batch))

            # Reset the batch
            batch = []
            uttids = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=Path,
                        default="/home/cxiao7/research/sssd/to_label/16000/annotations/segments/segments",
                        help="Path to the keyfile")
    parser.add_argument("--dset", type=Path,
                        default="/expscratch/dchakraborty/hf_datasets/scale23/data/multi/cmn.bbn_cts_bolt_test",
                        help="Path to the HF dataset")
    parser.add_argument("--src-lang", type=str, default="eng",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path,
                        default="/home/cxiao7/research/sssd/to_label/16000/decode/seamless",
                        help="Path to the output directory")
    parser.add_argument("--task", type=str, default="ASR",
                        choices=["ASR", "MT", "ST"],
                        help="Task to perform")
    parser.add_argument("--model-name", type=str, default="large")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Number of beams for the inference")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of the process for multi-GPU inference")

    args = parser.parse_args()
    inference(
        keyfile=args.keyfile,
        dset=args.dset,
        src_lang=LANGS[args.src_lang],
        tgt_lang=LANGS[args.tgt_lang],
        output_dir=args.output_dir,
        model_name=args.model_name,
        task=args.task,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        rank=args.rank,
    )


if __name__ == "__main__":
    main()