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

LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
    "tus": "tunisian",
    "eng": "english",
}


def inference(keyfile, dset, src_lang, tgt_lang, output_dir, model_name, pretrained_model=None, peft_model=None, task="transcribe", batch_size=1):
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
            print(f"Loading model from huggingface openai/whisper-{model_name}")
            processor = WhisperProcessor.from_pretrained(
                f"openai/whisper-{model_name}")
            model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/whisper-{model_name}").to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=src_lang, task=task)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    if task=="bmtl":
        output_asr = output_dir / "asr"
        output_st = output_dir / "st"
        startoftranslation_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranslation|>")
        with open(output_asr, "w") as f_asr, open(output_st, "w") as f_st:
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
                input_speech = [utt["audio"]["array"] for utt in batch]
                samping_rate = batch[0]["audio"]["sampling_rate"]
                input_features = processor(
                    input_speech, sampling_rate=samping_rate, return_tensors="pt").input_features.to(device)
                # Generate token ids
                predicted_ids = model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids)
                # Decode token ids to text
                # If the model fails to produce the startoftranslation token, ignore the whole sequence for ASR eval
                asr_hyp_ids = []
                st_hyp_ids = []
                for i, predicted_id in enumerate(predicted_ids):
                    try:
                        st_hyp_ids.append(predicted_id[predicted_id.tolist().index(startoftranslation_id):])
                        asr_hyp_ids.append(predicted_id[:predicted_id.tolist().index(startoftranslation_id)])
                    except ValueError:
                        asr_hyp_ids.append([])
                        st_hyp_ids.append(predicted_id)
                asr_hyps = processor.batch_decode(
                    asr_hyp_ids, skip_special_tokens=True)
                st_hyps = processor.batch_decode(
                    st_hyp_ids, skip_special_tokens=True)
                for i, (asr_hyp, st_hyp) in enumerate(zip(asr_hyps, st_hyps)):
                    print(uttids[i], asr_hyp, file=f_asr)
                    print(uttids[i], st_hyp, file=f_st)
                f_asr.flush()
                f_st.flush()
                pbar.update(len(batch))

                # Reset the batch
                batch = []
                uttids = []
    else:
        output = output_dir / "text"
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
                input_speech = [utt["audio"]["array"] for utt in batch]
                samping_rate = batch[0]["audio"]["sampling_rate"]
                input_features = processor(
                    input_speech, sampling_rate=samping_rate, return_tensors="pt").input_features.to(device)
                # Generate token ids
                predicted_ids = model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids)
                # Decode token ids to text
                hyps = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True)
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
    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate", "bmtl"],
                        help="Task to perform")
    parser.add_argument("--pretrained-model", type=Path, default=None,
                        help="Path to the pretrained (finetuned) model, if not specified, the model will be loaded from HuggingFace")
    parser.add_argument("--peft-model", type=Path, default=None,
                        help="Path to the PEFT model, note that this will override the pretrained model")
    parser.add_argument("--model_name", type=str, default="tiny")
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()
    inference(keyfile=args.keyfile,
              dset=args.dset,
              src_lang=LANGS[args.src_lang],
              tgt_lang=LANGS[args.tgt_lang],
              output_dir=args.output_dir,
              model_name=args.model_name,
              task=args.task,
              pretrained_model=args.pretrained_model,
              peft_model=args.peft_model,
              batch_size=args.batch_size)


if __name__ == "__main__":
    main()
