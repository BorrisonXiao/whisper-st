#!/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
# Note that the hard-coded path above is specific to the HLT cluster due to ESPNet environment setup.
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from pathlib import Path
from peft import get_peft_model, LoraConfig
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from pyscripts.utils.config_parser import Config
import logging
import json
from functools import partial

LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
    "eng": "english",
}


def load_train_and_dev_sets(hf_datadir, train_set, src_lang, tgt_lang, mode="asr", dev_name="dev"):
    src_langs = [src_lang] if src_lang != "all" else list(LANGS.keys())
    if src_lang == "all":
        # For multilingual training, we need to load all the datasets
        raise NotImplementedError
    train_dset_dict = {}
    val_dset_dict = {}
    for lang in src_langs:
        if mode == "mtl":
            # For multi-task learning, the dataset will be duplicated and concatenated
            # with the other task's dataset. Note that all both the transcript and
            # translation will be named as "text" in the returned dataset.
            raise NotImplementedError
        elif mode == "asr":
            # For ASR, we only need the transcript
            train_dset = load_from_disk(hf_datadir / f"{lang}.{train_set}")
            val_dset = load_from_disk(hf_datadir / f"{lang}.{dev_name}")
            # Rename the "transcript" column to "text"
            train_dset = train_dset.rename_column("transcript", "text")
            val_dset = val_dset.rename_column("transcript", "text")
            # Remove the "translation" column
            train_dset = train_dset.remove_columns([col for col in train_dset.column_names if col not in [
                                                   "audio", "text", "src_lang", "tgt_lang"]])
            val_dset = val_dset.remove_columns([col for col in val_dset.column_names if col not in [
                                               "audio", "text", "src_lang", "tgt_lang"]])
            # TODO: For ASR, it might be better to convert tgt_lang to src_lang.
            # Not doing this for now due to permission issues.
            # train_dset = train_dset.map(lambda x: {"tgt_lang": x["src_lang"]})
            # val_dset = val_dset.map(lambda x: {"tgt_lang": x["src_lang"]})
            train_dset_dict[f"{src_lang}_{mode}"] = train_dset
            val_dset_dict[f"{src_lang}_{mode}"] = val_dset
        elif mode == "st":
            # For ST, we need only the  translation
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return train_dset_dict, val_dset_dict


def prepare_dataset(
        dset_dict,
        model_name,
        local_rank,
        preprocessing_num_proc=4,
        normalize_text=False,
        save_feature_dir=None,
        dset_type="train",
        on_the_fly_feat_extraction=False,
        train=True,
):
    processed_dset_list = []
    for _lang, dset in dset_dict.items():
        lang, mode = _lang.split("_")
        # If the features are already extracted, load them directly
        if save_feature_dir is not None and (save_feature_dir / f"{lang}.{dset_type}.{mode}").exists():
            if not train:
                # If feature extraction only, no need to return anything
                return None
            logging.info(
                f"Found precomputed features, loading from {save_feature_dir / f'{lang}.{dset_type}.{mode}'}")
            processed_dset = load_from_disk(
                save_feature_dir / f"{lang}.{dset_type}.{mode}")
            processed_dset_list.append(processed_dset)
        else:
            if save_feature_dir is not None:
                _load_file_dir = save_feature_dir / \
                    f"{lang}.{dset_type}.{mode}"
                logging.warning(
                    f"Feature directory {_load_file_dir} does not exist, extracting features...")
            if mode == "asr":
                std = BasicTextNormalizer() if lang != "eng" else EnglishTextNormalizer()
            else:
                std = EnglishTextNormalizer()
            processor = WhisperProcessor.from_pretrained(
                f"openai/whisper-{model_name}", language=LANGS[lang], task="transcribe" if mode == "asr" else "translate")

            def _prepare_dataset(batch):
                audio = batch["audio"]
                if not on_the_fly_feat_extraction:
                    batch["input_features"] = processor.feature_extractor(
                        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
                text = std(batch["text"]).strip(
                ) if normalize_text else batch["text"]
                batch["labels"] = processor.tokenizer(text).input_ids
                return batch

            col_names = dset.column_names if not on_the_fly_feat_extraction else [
                col for col in dset.column_names if col != "audio"]
            processed_dset = dset.map(_prepare_dataset,
                                      num_proc=preprocessing_num_proc,
                                      remove_columns=col_names,
                                      desc="Preprocessing dataset")
            if local_rank in [-1, 0] and save_feature_dir is not None:
                if (save_feature_dir / f"{lang}.{dset_type}.{mode}").exists():
                    logging.warning(
                        f"Feature directory {save_feature_dir / f'{lang}.{dset_type}.{mode}'} already exists, skipping...")
                else:
                    processed_dset.save_to_disk(
                        save_feature_dir / f"{lang}.{dset_type}.{mode}")
            processed_dset_list.append(processed_dset)

    # Concatenate all the datasets
    return concatenate_datasets(processed_dset_list)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    This portion of code is adapted from:
    https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if "input_features" not in features[0]:
            # Perform feature extraction on the fly
            input_features = [{"input_features": self.processor.feature_extractor(
                feature["audio"]["array"], sampling_rate=feature["audio"]["sampling_rate"]).input_features[0]} for feature in features]
        else:
            input_features = [{"input_features": feature["input_features"]}
                              for feature in features]
        # Convert to tensors
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # Pad label ids to the max length in the batch
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        _bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>")
        if (labels[:, 0] == _bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # breakpoint()
        return batch


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
                metric_for_best_model="cer",
                greater_is_better=False,
                remove_unused_columns=False,  # This is important for PEFT
            ),
            "LoraConfig": dict(
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",
            )}
    else:
        res = Config(config).get()
        res["Seq2SeqTrainingArguments"]["learning_rate"] = float(
            res["Seq2SeqTrainingArguments"]["learning_rate"])
        res["Seq2SeqTrainingArguments"]["weight_decay"] = float(
            res["Seq2SeqTrainingArguments"]["weight_decay"])
        return res


def feat_extraction(
    hf_datadir,
    train_set,
    src_lang,
    tgt_lang,
    model_name,
    mode="asr",
    preprocessing_num_proc=4,
    normalize_text=False,
    save_feature_dir=None,
    local_rank=-1,
    on_the_fly_feat_extraction=False,
    dev_name="dev",
    train=True,
):
    # Step 1: Load the sets
    train_dset_dict, val_dset_dict = load_train_and_dev_sets(
        hf_datadir, train_set, src_lang, tgt_lang, mode=mode, dev_name=dev_name)

    # Step 2: Data augmentation

    # Step 3: Feature extraction
    _train_dset = prepare_dataset(dset_dict=train_dset_dict,
                                  model_name=model_name,
                                  preprocessing_num_proc=preprocessing_num_proc,
                                  normalize_text=normalize_text,
                                  save_feature_dir=save_feature_dir,
                                  dset_type="train",
                                  local_rank=local_rank,
                                  on_the_fly_feat_extraction=on_the_fly_feat_extraction,
                                  train=train)
    _val_dset = prepare_dataset(dset_dict=val_dset_dict,
                                model_name=model_name,
                                preprocessing_num_proc=preprocessing_num_proc,
                                normalize_text=normalize_text,
                                save_feature_dir=save_feature_dir,
                                dset_type="dev",
                                local_rank=local_rank,
                                on_the_fly_feat_extraction=on_the_fly_feat_extraction,
                                train=train)
    # breakpoint()

    return _train_dset, _val_dset


def finetune(
    train_set,
    hf_datadir,
    src_lang,
    tgt_lang,
    output_dir,
    model_name,
    local_rank,
    config=None,
    mode="asr",
    preprocessing_num_proc=4,
    normalize_text=False,
    save_feature_dir=None,
    load_model_from_path=None,
    resume_from_checkpoint=None,
    peft_method=None,
    on_the_fly_feat_extraction=False,
    dev_name="dev",
    deepspeed=None,
    save_eval_preds=None,
):
    # Step 1: Load prepare the training/dev sets
    _train_dset, _val_dset = feat_extraction(
        hf_datadir=hf_datadir,
        train_set=train_set,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model_name=model_name,
        mode=mode,
        preprocessing_num_proc=preprocessing_num_proc,
        normalize_text=normalize_text,
        save_feature_dir=save_feature_dir,
        local_rank=local_rank,
        on_the_fly_feat_extraction=on_the_fly_feat_extraction,
        dev_name=dev_name,
        train=True,
    )

    # Step 4: Define the data collator
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{model_name}")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Step 5: Define the metric
    metric = evaluate.load("cer")

    def compute_metrics(pred, normalize_eval=False):
        # breakpoint()
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        pred_str_raw = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=False)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)

        if normalize_eval:
            std = BasicTextNormalizer()
            pred_str = [std(pred) for pred in pred_str]
            label_str = [std(label) for label in label_str]
            # Filtering step to only evaluate the samples that correspond to non-zero references
            pred_str = [pred_str[i]
                        for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i]
                         for i in range(len(label_str)) if len(label_str[i]) > 0]

        if save_eval_preds is not None:
            with open(save_eval_preds, "a") as f:
                for i, (pred, label) in enumerate(list(zip(pred_str, label_str))[:1000]):
                    f.write(f"Ref: {label}\n")
                    f.write(f"Pred: {''.join(pred_str_raw[i].split()[:4])}{pred}\n")
                print("--------------------------------------------------", file=f)

        cer = metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # TODO (Cihan): Add parameter overrides
    _args = parse_config(config)
    _training_args = _args['Seq2SeqTrainingArguments']
    # Load the deepspeed config if specified
    if deepspeed is not None:
        with open(deepspeed, "r") as f:
            ds_config_dict = json.load(f)
        _training_args["deepspeed"] = ds_config_dict
    logging.info(f"Training Config: {_training_args}")

    # Step 6: Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, **_training_args)

    # Step 7: Load the model and trainer
    if load_model_from_path:
        model = WhisperForConditionalGeneration.from_pretrained(
            load_model_from_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{model_name}")

    if resume_from_checkpoint and peft_method is None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model = model.from_pretrained(resume_from_checkpoint)

    if peft_method:
        if peft_method == "lora":
            _peft_config = _args['LoraConfig']
            peft_config = LoraConfig(
                inference_mode=False,
                **_peft_config
            )
            logging.info(f"PEFT Config: {peft_config}")
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            training_args.output_dir = str(output_dir)
        elif peft_method == "qlora":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown PEFT method: {peft_method}")

    # TODO: Extend this for multilingual training
    model.generate = partial(
        model.generate, language=LANGS[src_lang], task="transcribe" if mode == "asr" else "translate")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=_train_dset,
        eval_dataset=_val_dset,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Step 8: Launch training
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint)
    else:
        trainer.train()

    # Step 9: Save the model
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to the config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO",
                                 "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--train-set", type=str,
                        default="train-cts_sp",
                        help="Name of the training set")
    parser.add_argument("--hf_datadir", type=Path,
                        default="/expscratch/dchakraborty/hf_datasets/scale23/data/all",
                        help="Path to the HF datasets")
    parser.add_argument("--src-lang", type=str, default="cmn",
                        help="Source language")
    parser.add_argument("--tgt-lang", type=str, default="eng",
                        help="Target language")
    parser.add_argument("--output_dir", type=Path,
                        default="ft_exp/hf_whisper_tiny/cmn/asr/",
                        help="Path to the output directory")
    parser.add_argument("--mode", type=str, default="asr",
                        choices=["asr", "st", "mtl"],
                        help="Task to perform")
    parser.add_argument("--preprocessing_num_proc", type=int, default=4,
                        help="Number of processes to use for preprocessing")
    parser.add_argument("--normalize_text", action="store_true",
                        help="Whether to normalize the text")
    parser.add_argument("--save_feature_dir", type=Path,
                        default="/exp/cxiao/scale23/hf_data/features",
                        help="Path to the directory to save the extracted features, if None, the features will not be saved")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--load_model_from_path", type=Path, default=None,
                        help="Path to the model to load")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None,
                        help="Path to the checkpoint to resume from, note that this overrides the load_model_from_path option")
    parser.add_argument("--on-the-fly-feat-extraction", action="store_true",
                        help="Whether to extract features on the fly without saving them")
    parser.add_argument("--peft_method", type=str, default=None,
                        choices=["lora", "qlora"],
                        help="Which PEFT method to use")
    parser.add_argument("--dev-name", type=str, default="dev",
                        choices=["dev", "dev1", "dev2"],
                        help="Name of the dev set, e.g. dev, dev1, dev2")
    parser.add_argument("--feat-extraction", action="store_true",
                        help="If true, only perform feature extraction")
    parser.add_argument("--deepspeed", type=Path, default=None,
                        help="Path to the deepspeed config file")
    parser.add_argument("--save-eval-preds", type=Path, default=None,
                        help="Path to the file to save the validation predictions, if None, the predictions will not be saved")
    parser.add_argument("--model_name", type=str, default="tiny")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    if args.feat_extraction:
        feat_extraction(
            train_set=args.train_set,
            hf_datadir=args.hf_datadir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            model_name=args.model_name,
            mode=args.mode,
            preprocessing_num_proc=args.preprocessing_num_proc,
            save_feature_dir=args.save_feature_dir,
            local_rank=args.local_rank,
            normalize_text=args.normalize_text,
            on_the_fly_feat_extraction=args.on_the_fly_feat_extraction,
            dev_name=args.dev_name,
            train=False,
        )
    else:
        finetune(
            config=args.config,
            train_set=args.train_set,
            hf_datadir=args.hf_datadir,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            output_dir=args.output_dir,
            model_name=args.model_name,
            mode=args.mode,
            preprocessing_num_proc=args.preprocessing_num_proc,
            save_feature_dir=args.save_feature_dir,
            local_rank=args.local_rank,
            normalize_text=args.normalize_text,
            load_model_from_path=args.load_model_from_path,
            resume_from_checkpoint=args.resume_from_checkpoint,
            peft_method=args.peft_method,
            on_the_fly_feat_extraction=args.on_the_fly_feat_extraction,
            deepspeed=args.deepspeed,
            dev_name=args.dev_name,
            save_eval_preds=args.save_eval_preds,
        )


if __name__ == "__main__":
    main()
