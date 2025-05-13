#!/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
# Note that the hard-coded path above is specific to the HLT cluster due to ESPNet environment setup.
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import Dataset, Features, Value
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, prepare_model_for_kbit_training
import logging
import evaluate
from pyscripts.utils.config_parser import Config
import json


LANGS = {
    "cmn": "zho_Hans",
    "spa": "spa_Latn",
    "rus": "rus_Cyrl",
    "eng": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
}

def parse_config(config):
    if config is None:
        # The default config
        return {
            "Seq2SeqTrainingArguments": dict(
                per_device_train_batch_size=16,
                per_device_eval_batch_size=24,
                gradient_accumulation_steps=1,
                warmup_steps=800,
                max_steps=2400,
                learning_rate=1e-4,
                weight_decay=0.01,
                fp16=True,
                predict_with_generate=True,
                generation_max_length=225,
                logging_steps=30,
                report_to=["tensorboard"],
                evaluation_strategy="steps",
                eval_steps=1,
                save_strategy="steps",
                save_steps=300,
                load_best_model_at_end=True,
                metric_for_best_model="sacrebleu",
                greater_is_better=False,
                remove_unused_columns=False,
            ),
            "LoraConfig": dict(
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",
            ),
        }
    else:
        res = Config(config).get()
        res["Seq2SeqTrainingArguments"]["learning_rate"] = float(
            res["Seq2SeqTrainingArguments"]["learning_rate"])
        res["Seq2SeqTrainingArguments"]["weight_decay"] = float(
            res["Seq2SeqTrainingArguments"]["weight_decay"])

        if "QuantizationConfig" in res:
            if "bnb_4bit_compute_dtype" in res['QuantizationConfig']:
                if res['QuantizationConfig']['bnb_4bit_compute_dtype'] == "bf16":
                    res['QuantizationConfig']['bnb_4bit_compute_dtype'] = torch.bfloat16
                elif res['QuantizationConfig']['bnb_4bit_compute_dtype'] == "fp16":
                    res['QuantizationConfig']['bnb_4bit_compute_dtype'] = torch.float16
                else:
                    raise ValueError(
                        f"Unsupported compute dtype: {res['QuantizationConfig']['bnb_4bit_compute_dtype']}")

        return res


def create_dataset_stm(
    input_dir: Path,
    dset: str,
    src_lang: str,
    tgt_lang: str,
    output_dir: Path = None,
):
    # Read the text files
    src_text = input_dir / dset / f"text.tc.{src_lang}"
    tgt_text = input_dir / dset / f"text.tc.{tgt_lang}"

    with open(src_text, "r") as f:
        src_lines = f.readlines()
    with open(tgt_text, "r") as f:
        tgt_lines = f.readlines()

    transcripts = []
    uttids = []
    translations = []
    src_langs = [src_lang for _ in range(len(src_lines))]
    tgt_langs = [tgt_lang for _ in range(len(tgt_lines))]
    for src_line, tgt_line in zip(src_lines, tgt_lines):
        uttid, src_text = src_line.strip().split(maxsplit=1)
        uttid_tgt, tgt_text = tgt_line.strip().split(maxsplit=1)
        assert uttid == uttid_tgt, f"uttid mismatch: {uttid} != {uttid_tgt}"
        transcripts.append(src_text)
        uttids.append(uttid)
        translations.append(tgt_text)

    # Create the dataset
    features = Features({
        "uttid": Value(dtype="string"),
        "transcript": Value(dtype="string"),
        "translation": Value(dtype="string"),
        "src_lang": Value(dtype="string"),
        "tgt_lang": Value(dtype="string"),
    })

    dataset = Dataset.from_dict({"uttid": uttids, "transcript": transcripts, "translation": translations,
                                "src_lang": src_langs, "tgt_lang": tgt_langs}, features=features)

    if output_dir is not None:
        dataset.save_to_disk(output_dir)

    return dataset


def finetune(
    config,
    input_dir,
    hf_datadir,
    train_set,
    src_lang,
    tgt_lang,
    output_dir,
    preprocessing_num_proc,
    local_rank,
    load_model_from_path,
    resume_from_checkpoint,
    peft_method,
    dev_set,
    deepspeed,
    model_name,
):
    train_dset = create_dataset_stm(
        input_dir=input_dir,
        dset=train_set,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_dir=hf_datadir,
    )

    dev_dset = create_dataset_stm(
        input_dir=input_dir,
        dset=dev_set,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        output_dir=hf_datadir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        f"facebook/nllb-200-{model_name}",
        src_lang=LANGS[src_lang],
        tgt_lang=LANGS[tgt_lang],
    )
    
    def preprocess_func(
        examples,
    ):
        return tokenizer(
            text=examples["transcript"],
            text_target=examples["translation"],
            max_length=512,
            truncation=True,
        )
    
    col_names = [col for col in train_dset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
    # Tokenize the dataset
    train_dset = train_dset.map(
        preprocess_func,
        batched=True,
        batch_size=8,
        num_proc=preprocessing_num_proc,
        remove_columns=col_names,
        desc="Tokenizing the training set",
    )
    dev_dset = dev_dset.map(
        preprocess_func,
        batched=True,
        batch_size=8,
        num_proc=preprocessing_num_proc,
        remove_columns=col_names,
        desc="Tokenizing the validation set",
        
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    _peft = peft_method if peft_method is not None else "none"
    experiment_id = f"nllb_{model_name}_{src_lang}_{_peft}_{train_set}"
    metric_sacrebleu = evaluate.load("sacrebleu", experiment_id=experiment_id)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)

        # Use sacrebleu for translation training
        sacrebleu = metric_sacrebleu.compute(
            predictions=pred_str, references=label_str)
        return {"sacrebleu": sacrebleu['score']}

    _args = parse_config(config)
    _training_args = _args['Seq2SeqTrainingArguments']
    # Load the deepspeed config if specified
    if deepspeed is not None:
        with open(deepspeed, "r") as f:
            ds_config_dict = json.load(f)
        _training_args["deepspeed"] = ds_config_dict
    logging.info(f"Training Config: {_training_args}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, **_training_args)

    # Step 6.1: Parse the quantiation arguments if qlora is used
    quantization_config = None
    load_in_4bit = False
    load_in_8bit = False
    if peft_method and peft_method == "qlora":
        _quantization_config = _args.get('QuantizationConfig', None)
        if _quantization_config is None:
            _quantization_config = dict(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_compute_dtype="bf16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        quantization_config = BitsAndBytesConfig(
            **_quantization_config
        )
        assert not (
            load_in_4bit and load_in_8bit), "Cannot load in both 4bit and 8bit"
        logging.info(f"Quantization Config: {quantization_config}")

    # Step 7: Load the model and trainer
    if load_model_from_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            load_model_from_path,
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            f"facebook/nllb-200-{model_name}",
            quantization_config=quantization_config,
        )

    if resume_from_checkpoint and peft_method is None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model = model.from_pretrained(
            resume_from_checkpoint,
            quantization_config=quantization_config,
        )

    if peft_method:
        if peft_method == "lora" or peft_method == "qlora":
            _peft_config = _args.get("LoraConfig", {})
            peft_config = LoraConfig(
                inference_mode=False,
                **_peft_config,
            )
            logging.info(f"PEFT Config: {peft_config}")
            if peft_method == "qlora":
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            training_args.output_dir = str(output_dir)
        else:
            raise ValueError(f"Unknown PEFT method: {peft_method}")

    trainer = Seq2SeqTrainer(model=model,args=training_args,data_collator=data_collator,compute_metrics=compute_metrics,train_dataset=train_dset,eval_dataset=dev_dset,)
    
    # Step 8: Launch training
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint)
    else:
        trainer.train()
        
    # Step 9: Save the model
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to the config file")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Path to the input directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO",
                                 "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--train-set", type=str, required=True,
                        help="Training set name")
    parser.add_argument("--hf-datadir", type=Path, default=None,
                        help="Path to the Hugging Face dataset directory")
    parser.add_argument("--src-lang", type=str, default="cmn",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path,
                        default="ft_exp/hf_whisper_tiny/cmn/asr/",
                        help="Path to the output directory")
    parser.add_argument("--preprocessing_num_proc", type=int, default=4,
                        help="Number of processes to use for preprocessing")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--load_model_from_path", type=Path, default=None,
                        help="Path to the model to load")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None,
                        help="Path to the checkpoint to resume from, note that this overrides the load_model_from_path option")
    parser.add_argument("--peft_method", type=str, default=None,
                        choices=["lora", "qlora", None],
                        help="Which PEFT method to use")
    parser.add_argument("--dev-set", type=str, required=True,
                        help="Name of the dev set, e.g. dev, dev1, dev2")
    parser.add_argument("--deepspeed", type=Path, default=None,
                        help="Path to the deepspeed config file")
    parser.add_argument("--model_name", type=str, default="1.3B")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    finetune(
        input_dir=args.input_dir,
        config=args.config,
        train_set=args.train_set,
        hf_datadir=args.hf_datadir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        output_dir=args.output_dir,
        preprocessing_num_proc=args.preprocessing_num_proc,
        local_rank=args.local_rank,
        load_model_from_path=args.load_model_from_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        peft_method=args.peft_method,
        dev_set=args.dev_set,
        deepspeed=args.deepspeed,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
