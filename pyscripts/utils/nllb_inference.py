#!/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
# Note that the hard-coded path above is specific to the HLT cluster due to ESPNet environment setup.
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_from_disk
from peft import PeftModel, PeftConfig

LANGS = {
    "cmn": "zho_Hans",
    "spa": "spa_Latn",
    "rus": "rus_Cyrl",
    "eng": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
}


def inference(
    keyfile,
    src_lang,
    tgt_lang,
    output_dir,
    model_name,
    pretrained_model=None,
    peft_model=None,
    batch_size=1,
    num_beams=2,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    if peft_model is not None:
        print(f"Loading model from {peft_model}")
        peft_config = PeftConfig.from_pretrained(peft_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            peft_config.base_model_name_or_path).to(device)
        processor = AutoTokenizer.from_pretrained(peft_model)
        model = PeftModel.from_pretrained(model, peft_model)
    else:
        if pretrained_model is not None:
            print(f"Loading model from {pretrained_model}")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model).to(device)
        else:
            print(
                f"Loading model from huggingface facebook/nllb-200-{model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                f"facebook/nllb-200-{model_name}",
                src_lang=src_lang,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                f"facebook/nllb-200-{model_name}").to(device)

    print(f"model.device: {model.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"batch_size: {batch_size}")
    print(f"num_beams: {num_beams}")

    data_list = []
    if keyfile.suffix == ".text":
        with open(keyfile, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines):
            uttid, text = line.strip().split(maxsplit=1)
            data_list.append({"uttid": uttid, "text": text})
    else:
        raise NotImplementedError()

    total_len = len(data_list)
    pbar = tqdm(range(total_len))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "text"
    with open(output, "w") as f:
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            uttids = [d["uttid"] for d in batch]
            texts = [d["text"] for d in batch]
            inputs = tokenizer(texts, return_tensors="pt",
                               padding=True, truncation=True).to(device)
            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(f"{tgt_lang}"), max_length=300, num_beams=num_beams
            )
            hyps = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True)
            for j, hyp in enumerate(hyps):
                print(f"{uttids[j]} {hyp}", file=f)
            f.flush()
            pbar.update(len(batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=Path, required=True,
                        help="Path to the keyfile")
    parser.add_argument("--src-lang", type=str, default="cmn",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to the output directory")
    parser.add_argument("--pretrained-model", type=Path, default=None,
                        help="Path to the pretrained (finetuned) model, if not specified, the model will be loaded from HuggingFace")
    parser.add_argument("--peft-model", type=Path, default=None,
                        help="Path to the PEFT model, note that this will override the pretrained model")
    parser.add_argument("--model_name", type=str, default="1.3B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=2,
                        help="Number of beams for the inference")

    args = parser.parse_args()
    inference(keyfile=args.keyfile,
              src_lang=LANGS[args.src_lang],
              tgt_lang=LANGS[args.tgt_lang],
              output_dir=args.output_dir,
              model_name=args.model_name,
              pretrained_model=args.pretrained_model,
              peft_model=args.peft_model,
              batch_size=args.batch_size,
              num_beams=args.num_beams)


if __name__ == "__main__":
    main()
