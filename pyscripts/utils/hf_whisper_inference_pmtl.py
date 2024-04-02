#!/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
# Note that the hard-coded path above is specific to the HLT cluster due to ESPNet environment setup.
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

# For debugging ESPNet env issue (not loading the correct python interpreter)
# for dist in __import__('pkg_resources').working_set:
#     print(dist.project_name.replace('Python', ''))
# import sys; print(sys.executable)

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_from_disk
from peft import PeftModel, PeftConfig
from functools import partial
import re

LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
    "tus": "tunisian",
    "eng": "english",
}


def inference(
    keyfile,
    dset,
    src_lang,
    tgt_lang,
    output_dir,
    model_name,
    pretrained_model=None,
    peft_model=None,
    batch_size=1,
    use_asr_hyp=False,
    disable_asr=False,
    num_beams=2,
    ):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    if peft_model is not None:
        print(f"Loading model from {peft_model}")
        peft_config = PeftConfig.from_pretrained(peft_model)
        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path).to(device)
        processor = WhisperProcessor.from_pretrained(peft_model)
        model = PeftModel.from_pretrained(model, peft_model)
    else:
        if pretrained_model is not None:
            print(f"Loading model from {pretrained_model}")
            processor = WhisperProcessor.from_pretrained(pretrained_model)
            model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model).to(device)
        else:
            print(
                f"Loading model from huggingface openai/whisper-{model_name}")
            processor = WhisperProcessor.from_pretrained(
                f"openai/whisper-{model_name}")
            model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/whisper-{model_name}").to(device)

    asr_forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=src_lang, task="transcribe")
    print(f"model.device: {model.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

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

    startofprev_id = processor.tokenizer.convert_tokens_to_ids(
        "<|startofprev|>")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_asr = output_dir / "asr"
    output_st = output_dir / "st"
    do_asr = use_asr_hyp or not disable_asr
    if do_asr:
        f = open(output_asr, "w")
    else:
        print(f"Skipping ASR inference as use-asr-hyp is {use_asr_hyp} and disable-asr is {disable_asr}...")
    with open(output_st, "w") as f_st:
        total_len = len(ds) if keyfile is None else len(keys)
        pbar = tqdm(range(total_len))
        batch = []
        uttids = []
        for uttidx, utt in enumerate(ds):
            uttid = utt["uttid"]
            if keyfile is not None and uttid not in keys:
                continue
            # Accumulate the batch
            uttids.append(uttid)
            batch.append(utt)
            if len(batch) < batch_size and uttid != last_uttid:
                continue
            # Process the batch
            input_speech = [utt["audio"]["array"] for utt in batch]
            samping_rate = batch[0]["audio"]["sampling_rate"]
            input_features = processor(
                input_speech, sampling_rate=samping_rate, return_tensors="pt").input_features.to(device)
            
            if not do_asr:
                # Place holder if ASR is disabled
                hyps = ["" for _ in range(len(batch))]
            else:
                # Generate token ids
                model.generate = partial(
                    model.generate, language=src_lang, task="transcribe")
                predicted_ids = model.generate(
                    input_features, max_length=448, forced_decoder_ids=asr_forced_decoder_ids)
                # Decode token ids to text
                hyps = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True)
                    
            st_predicted_ids = []
            model.generate = partial(
                model.generate, language=src_lang, task="translate")
            for i, hyp in enumerate(hyps):
                inputs_st = {}
                inputs_st["input_features"] = input_features[i].unsqueeze(0)
                if use_asr_hyp:
                    prompt_ids = torch.Tensor([startofprev_id] + processor.tokenizer.encode(
                        hyp, add_special_tokens=False)).long().to(device)
                else:
                    # Normalize the reference text, remove things like "[xxxx]", i.e. things in square brackets
                    _ref = re.sub(r"\s+\[.*?\]\s+", "", batch[i]["transcript"])
                    # Remove the "% " tokens
                    _ref = re.sub(r"% ", "", _ref)
                    # Remove the "<char> - " tokens where <char> refers to any English character
                    _ref = re.sub(r"[a-zA-Z] -\s+", "", _ref)
                    _ref = re.sub(r" - ", "", _ref)
                    # Remove the extra spaces
                    _ref = re.sub(r"\s+", " ", _ref).strip()
                    prompt_ids = torch.Tensor([startofprev_id] + processor.tokenizer.encode(
                        _ref, add_special_tokens=False)).long().to(device)
                # Only batch size of 1 is supported as the prompted ids do not support batching
                st_predicted_id = model.generate(
                    **inputs_st, max_length=448, prompt_ids=prompt_ids, num_beams=num_beams)
                st_predicted_ids.append(st_predicted_id[0])
            st_hyps = processor.tokenizer.batch_decode(
                st_predicted_ids, skip_special_tokens=True)
            for i, hyp in enumerate(hyps):
                if do_asr:
                    print(uttids[i], hyp, file=f)
                print(uttids[i], st_hyps[i], file=f_st)
            if do_asr:
                f.flush()
            f_st.flush()
            pbar.update(len(batch))

            # Reset the batch
            batch = []
            uttids = []
            
        if do_asr:
            f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfile", type=Path,
                        default="/home/hltcoe/cxiao/scale23/st/dump/export/cmn.bbn_cts_bolt_test.wav.scp",
                        help="Path to the keyfile")
    parser.add_argument("--dset", type=Path,
                        default="/expscratch/dchakraborty/hf_datasets/scale23/data/multi/cmn.bbn_cts_bolt_test",
                        help="Path to the HF dataset")
    parser.add_argument("--src-lang", type=str, default="cmn",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path,
                        default="exp/st_hf_whisper_tiny/logdir/inference_asr/cmn/bbn_cts_bolt_test/output.1",
                        help="Path to the output directory")
    parser.add_argument("--pretrained-model", type=Path, default=None,
                        help="Path to the pretrained (finetuned) model, if not specified, the model will be loaded from HuggingFace")
    parser.add_argument("--peft-model", type=Path, default=None,
                        help="Path to the PEFT model, note that this will override the pretrained model")
    parser.add_argument("--disable-asr", action="store_true",
                        help="Disable ASR inference, note that this only applies if use-asr-hyp is False")
    parser.add_argument("--model_name", type=str, default="tiny")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-asr-hyp", action="store_true",
                        help="Use the ASR hypothesis as the prompt to the ST inference")
    parser.add_argument("--num-beams", type=int, default=2,
                        help="Number of beams for the inference")

    args = parser.parse_args()
    inference(keyfile=args.keyfile,
              dset=args.dset,
              src_lang=LANGS[args.src_lang],
              tgt_lang=LANGS[args.tgt_lang],
              output_dir=args.output_dir,
              model_name=args.model_name,
              pretrained_model=args.pretrained_model,
              peft_model=args.peft_model,
              batch_size=args.batch_size,
              use_asr_hyp=args.use_asr_hyp,
              disable_asr=args.disable_asr,
              num_beams=args.num_beams)


if __name__ == "__main__":
    main()
