#!/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
# Note that the hard-coded path above is specific to the HLT cluster due to ESPNet environment setup.
# Copyright 2023 Johns Hopkins University (Cihan Xiao)
# -*- coding: utf-8 -*-

from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, WhisperTrainer, BitsAndBytesConfig
import argparse
from pathlib import Path
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch
from datasets import load_from_disk, concatenate_datasets, load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from pyscripts.utils.config_parser import Config
import logging
import json
from functools import partial
import chinese_converter
import re
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.trainer_whisper import _schedule_dynamic_mtl_weight


LANGS = {
    "ara": "arabic",
    "kor": "korean",
    "cmn": "chinese",
    "spa": "spanish",
    "rus": "russian",
}
DIALECT = {
    "tus": {"src_lang": "ara", "dialect": "tunisian"},
}


def load_train_and_dev_sets(hf_datadir, train_set, src_lang, tgt_lang, mode="asr", dev_name="dev"):
    src_langs = [src_lang] if src_lang != "all" else list(LANGS.keys())
    train_dset_dict = {}
    val_dset_dict = {}
    for lang in src_langs:
        if mode == "mtl":
            # For multi-task learning, the dataset will be duplicated and concatenated
            # with the other task's dataset. Note that all both the transcript and
            # translation will be named as "text" in the returned dataset.
            raw_train_dset = load_from_disk(hf_datadir / f"{lang}.{train_set}")
            raw_val_dset = load_from_disk(hf_datadir / f"{lang}.{dev_name}")
            # Rename the "transcript" column to "text" for the ASR task
            asr_train_dset = raw_train_dset.rename_column("transcript", "text")
            # Rename the "translation" column to "text" for the ST task
            st_train_dset = raw_train_dset.rename_column("translation", "text")
            st_val_dset = raw_val_dset.rename_column("translation", "text")
            # Remove the "translation" column for the ASR task
            asr_train_dset = asr_train_dset.remove_columns([col for col in asr_train_dset.column_names if col not in [
                "audio", "text", "src_lang", "tgt_lang"]])
            # Remove the "transcript" column for the ST task
            st_train_dset = st_train_dset.remove_columns([col for col in st_train_dset.column_names if col not in [
                "audio", "text", "src_lang", "tgt_lang"]])
            st_val_dset = st_val_dset.remove_columns([col for col in st_val_dset.column_names if col not in [
                "audio", "text", "src_lang", "tgt_lang"]])
            train_dset_dict[f"{lang}_asr"] = asr_train_dset
            train_dset_dict[f"{lang}_st"] = st_train_dset
            val_dset_dict[f"{lang}_st"] = st_val_dset
        elif mode == "pmtl":
            # For prompted multi-task learning, the dataset will not be duplicated
            # nor concatenated. Instead, the transcript and translation will be
            # kept in separate columns.
            train_dset = load_from_disk(hf_datadir / f"{lang}.{train_set}")
            val_dset = load_from_disk(hf_datadir / f"{lang}.{dev_name}")
            train_dset = train_dset.remove_columns([col for col in train_dset.column_names if col not in [
                "audio", "transcript", "translation", "src_lang", "tgt_lang"]])
            # Note that the "transcript" column is kept in case we want to evaluate on ASR at training time
            val_dset = val_dset.remove_columns([col for col in val_dset.column_names if col not in [
                "audio", "transcript", "translation", "src_lang", "tgt_lang"]])
            train_dset_dict[f"{lang}_{mode}"] = train_dset
            val_dset_dict[f"{lang}_{mode}"] = val_dset
        elif mode == "asr":
            if lang == "eng":
                # Using librispeech-100 for debugging
                train_dset = load_dataset(
                    "librispeech_asr", "clean", split="train.100")
                val_dset = load_dataset(
                    "librispeech_asr", "clean", split="validation")
                train_dset = train_dset.remove_columns([col for col in train_dset.column_names if col not in [
                    "audio", "text"]])
                val_dset = val_dset.remove_columns([col for col in val_dset.column_names if col not in [
                    "audio", "text"]])
                train_dset_dict[f"{lang}_{mode}"] = train_dset
                val_dset_dict[f"{lang}_{mode}"] = val_dset
            else:
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
                train_dset_dict[f"{lang}_{mode}"] = train_dset
                val_dset_dict[f"{lang}_{mode}"] = val_dset
        elif mode == "st":
            # For ST, we only need the translation
            train_dset = load_from_disk(hf_datadir / f"{lang}.{train_set}")
            val_dset = load_from_disk(hf_datadir / f"{lang}.{dev_name}")
            # Rename the "translation" column to "text"
            train_dset = train_dset.rename_column("translation", "text")
            val_dset = val_dset.rename_column("translation", "text")
            # Remove the "translation" column
            train_dset = train_dset.remove_columns([col for col in train_dset.column_names if col not in [
                "audio", "text", "src_lang", "tgt_lang"]])
            val_dset = val_dset.remove_columns([col for col in val_dset.column_names if col not in [
                "audio", "text", "src_lang", "tgt_lang"]])
            train_dset_dict[f"{lang}_{mode}"] = train_dset
            val_dset_dict[f"{lang}_{mode}"] = val_dset
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
        multilingual=False,
        _processor=None,
):
    processed_dset_list = []
    processed_dset_dict = {}
    for _lang, dset in dset_dict.items():
        lang, mode = _lang.split("_")
        # If the features are already extracted, load them directly
        if save_feature_dir is not None and (save_feature_dir / f"{lang}.{dset_type}.{mode}").exists():
            if not train:
                # If feature extraction only, no need to return anything
                continue
            logging.info(
                f"Found precomputed features, loading from {save_feature_dir / f'{lang}.{dset_type}.{mode}'}")
            processed_dset = load_from_disk(
                save_feature_dir / f"{lang}.{dset_type}.{mode}")
            processed_dset_list.append(processed_dset)
            processed_dset_dict[_lang] = processed_dset
        else:
            if save_feature_dir is not None:
                _load_file_dir = save_feature_dir / \
                    f"{lang}.{dset_type}.{mode}"
                logging.warning(
                    f"Feature directory {_load_file_dir} does not exist, extracting features...")

            processor = WhisperProcessor.from_pretrained(
                f"openai/whisper-{model_name}", language=LANGS[lang], task="translate" if mode == "st" else "transcribe") if not _processor else _processor

            if mode == "pmtl":
                if normalize_text:
                    std_basic = BasicTextNormalizer()
                    std_eng = EnglishTextNormalizer({})
                processor_eng = WhisperProcessor.from_pretrained(
                    f"openai/whisper-{model_name}", language=LANGS[lang], task="translate")

            def _prepare_dataset(batch):
                audio = batch["audio"]
                if not on_the_fly_feat_extraction:
                    batch["input_features"] = processor.feature_extractor(
                        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

                if mode == "pmtl":
                    scp = std_basic(batch["transcript"]).strip(
                    ) if normalize_text else batch["transcript"]
                    # if "train" in dset_type:
                    if True:
                        translation = std_eng(batch["translation"]).strip(
                        ) if normalize_text else batch["translation"]
                        batch["labels_src"] = processor.tokenizer(
                            scp).input_ids
                        batch["labels_tgt"] = processor_eng.tokenizer(
                            translation).input_ids
                        batch["labels_src_length"] = len(batch["labels_src"])
                        batch["labels_tgt_length"] = len(batch["labels_tgt"])
                    # else:
                    #     translation = std_eng(batch["text"]).strip(
                    #     ) if normalize_text else batch["text"]
                    #     batch["labels"] = processor_eng.tokenizer(
                    #         translation).input_ids
                    #     batch["labels_length"] = len(batch["labels"])
                else:
                    if mode == "asr":
                        std = BasicTextNormalizer() if lang != "eng" else EnglishTextNormalizer({})
                    else:
                        std = EnglishTextNormalizer({})
                    text = std(batch["text"]).strip(
                    ) if normalize_text else batch["text"]
                    batch["labels"] = processor.tokenizer(text).input_ids
                    batch["labels_length"] = len(batch["labels"])
                return batch

            col_names = dset.column_names if not on_the_fly_feat_extraction else [
                col for col in dset.column_names if col != "audio"]
            processed_dset = dset.map(_prepare_dataset,
                                      num_proc=preprocessing_num_proc,
                                      remove_columns=col_names,
                                      desc="Preprocessing dataset")
            # Filter out utterances whose token length exceeds 448
            max_label_length = 448  # 448 is the max length of the label sequence

            def filter_labels(labels_length):
                """Filter label sequences longer than max length (448)"""
                return labels_length < max_label_length

            if mode == "pmtl":
                processed_dset = processed_dset.filter(filter_labels, input_columns=[
                                                       "labels_src_length"])
                processed_dset = processed_dset.filter(filter_labels, input_columns=[
                                                       "labels_tgt_length"])
            else:
                processed_dset = processed_dset.filter(filter_labels, input_columns=[
                    "labels_length"])

            if local_rank in [-1, 0] and save_feature_dir is not None:
                if (save_feature_dir / f"{lang}.{dset_type}.{mode}").exists():
                    logging.warning(
                        f"Feature directory {save_feature_dir / f'{lang}.{dset_type}.{mode}'} already exists, skipping...")
                else:
                    processed_dset.save_to_disk(
                        save_feature_dir / f"{lang}.{dset_type}.{mode}")
            processed_dset_list.append(processed_dset)
            processed_dset_dict[_lang] = processed_dset

    # Concatenate all the datasets if training and not multilingual
    if "train" in dset_type or not multilingual:
        return concatenate_datasets(processed_dset_list) if len(processed_dset_list) > 0 else None
    return processed_dset_dict if len(processed_dset_list) > 0 else None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    This portion of code is adapted from:
    https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862
    """
    processor: Any
    dialects: dict = None

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

        asr_ref_prompt = False
        if "labels_src" in features[0]:
            # Flag indicating using ASR reference prompt
            asr_ref_prompt = True
        if asr_ref_prompt:
            label_features_src = [{"input_ids": feature["labels_src"]}
                                  for feature in features]
            label_features_tgt = [{"input_ids": feature["labels_tgt"]}
                                  for feature in features]

            labels_batch_src = self.processor.tokenizer.pad(
                label_features_src, return_tensors="pt")
            labels_batch_tgt = self.processor.tokenizer.pad(
                label_features_tgt, return_tensors="pt")

            labels_src = labels_batch_src["input_ids"].masked_fill(
                labels_batch_src.attention_mask.ne(1), -100)
            labels_tgt = labels_batch_tgt["input_ids"].masked_fill(
                labels_batch_tgt.attention_mask.ne(1), -100)

            _bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|startoftranscript|>")
            if (labels_src[:, 0] == _bos_token_id).all().cpu().item():
                labels_src = labels_src[:, 1:]
            if (labels_tgt[:, 0] == _bos_token_id).all().cpu().item():
                labels_tgt = labels_tgt[:, 1:]

            batch["labels_src"] = labels_src
            batch["labels_tgt"] = labels_tgt
        else:
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

            # If dialects are provided, replace the language id with the dialect id
            if self.dialects is not None and len(self.dialects) > 0:
                for src_lang, dialect in self.dialects.items():
                    src_lang_token = TO_LANGUAGE_CODE[src_lang]
                    src_lang_id = self.processor.tokenizer.convert_tokens_to_ids(
                        f"<|{src_lang_token}|>")
                    dialect_lang_token = TO_LANGUAGE_CODE[dialect]
                    dialect_lang_id = self.processor.tokenizer.convert_tokens_to_ids(
                        f"<|{dialect_lang_token}|>")
                    # Replace src_lang_id with dialect_lang_id
                    labels = labels.masked_fill(
                        labels.eq(src_lang_id), dialect_lang_id)

            batch["labels"] = labels

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
            ),
            "WhisperConfig": dict(
                apply_spec_augment=True,
                mask_time_prob=0.05,
                mask_time_length=10,
                mask_time_min_masks=2,
                mask_feature_prob=0.05,
                mask_feature_length=10,
                mask_feature_min_masks=0,
            )}
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

    # Step 2: Feature extraction
    # TODO (Cihan): Add support for subset multilingual training
    multilingual = True if src_lang == "all" else False
    _train_dset = prepare_dataset(dset_dict=train_dset_dict,
                                  model_name=model_name,
                                  preprocessing_num_proc=preprocessing_num_proc,
                                  normalize_text=normalize_text,
                                  save_feature_dir=save_feature_dir,
                                  dset_type=train_set,
                                  local_rank=local_rank,
                                  on_the_fly_feat_extraction=on_the_fly_feat_extraction,
                                  train=train,
                                  multilingual=multilingual,
                                  )
    _val_dset = prepare_dataset(dset_dict=val_dset_dict,
                                model_name=model_name,
                                preprocessing_num_proc=preprocessing_num_proc,
                                normalize_text=normalize_text,
                                save_feature_dir=save_feature_dir,
                                dset_type=dev_name,
                                local_rank=local_rank,
                                on_the_fly_feat_extraction=on_the_fly_feat_extraction,
                                train=train,
                                multilingual=multilingual,
                                )

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
    dialect=None,
    use_asr_prompt=False,
    min_promptless_prob=0.0,
    max_promptless_prob=0.0,
    max_sample_prob=0.0,
    min_sample_prob=0.0,
    min_alpha=0.5,
    max_alpha=0.5,
    loss_warmup='auto',
    loss_base=0.25,
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
    # If new languages are added, we need to add the corresponding language codes to the processor
    # TODO: Currently supports only one dialect
    dialects_dict = None
    if dialect is not None:
        special_tokens = processor.tokenizer.special_tokens_map
        lang_code = TO_LANGUAGE_CODE[DIALECT[dialect]['dialect']]
        lang_token = f"<|{lang_code}|>"
        special_tokens['additional_special_tokens'].append(lang_token)
        processor.tokenizer.add_special_tokens(special_tokens)
        # The dialects_dict is used to map the src_lang to the dialect name
        full_src_lang_name = LANGS[DIALECT[dialect]['src_lang']]
        dialects_dict = {full_src_lang_name: DIALECT[dialect]['dialect']}
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, dialects=dialects_dict)

    # Step 5: Define the metric
    _peft = peft_method if peft_method is not None else "none"
    experiment_id = f"{mode}_{model_name}_{src_lang}_{_peft}_{train_set}"
    metric_cer = evaluate.load("cer", experiment_id=experiment_id)
    metric_sacrebleu = evaluate.load("sacrebleu", experiment_id=experiment_id)

    def compute_metrics(pred, normalize_eval=False):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        # decode only the first 4 special tokens
        prefixes = processor.tokenizer.batch_decode(
            pred_ids[:, :4], skip_special_tokens=False)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)

        # Perform the traditional-to-simplified conversion for Chinese anyways
        # Also, due to the training set, some normalization is needed for Chinese
        if tgt_lang == "cmn":
            pred_str = [chinese_converter.to_simplified(
                char) for char in pred_str]
            label_str = [chinese_converter.to_simplified(
                char) for char in label_str]
            # Lowercase the English text for Chinese
            pred_str = [pred.lower() for pred in pred_str]
            label_str = [label.lower() for label in label_str]
            # Remove the punctuations in the Chinese text, including only ". ", ", ", "? ", "<O>% "
            pred_str = [re.sub(r"([.,?]|<O>% )", "", pred)
                        for pred in pred_str]
            label_str = [re.sub(r"([.,?]|<O>% )", "", label)
                         for label in label_str]

        if normalize_eval:
            std = BasicTextNormalizer() if tgt_lang != "eng" else EnglishTextNormalizer(
                {})
            pred_str = [std(pred) for pred in pred_str]
            label_str = [std(label) for label in label_str]
            # Filtering step to only evaluate the samples that correspond to non-zero references
            pred_str = [pred_str[i]
                        for i in range(len(pred_str)) if len(label_str[i]) > 0]
            label_str = [label_str[i]
                         for i in range(len(label_str)) if len(label_str[i]) > 0]

        if save_eval_preds is not None:
            Path(save_eval_preds).parent.mkdir(parents=True, exist_ok=True)
            with open(save_eval_preds, "a") as f:
                for i, (pred, label) in enumerate(list(zip(pred_str, label_str))[:200]):
                    f.write(f"Ref: {label}\n")
                    f.write(f"Pred: {pred}\n")
                    f.write(f"Prefix: {prefixes[i]}\n\n")
                print("--------------------------------------------------", file=f)

        if src_lang == tgt_lang:
            # Use CER for ASR training
            cer = metric_cer.compute(
                predictions=pred_str, references=label_str)
            return {"cer": cer}
        else:
            # Use sacrebleu for ST training
            sacrebleu = metric_sacrebleu.compute(
                predictions=pred_str, references=label_str)
            return {"sacrebleu": sacrebleu['score']}

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
        model = WhisperForConditionalGeneration.from_pretrained(
            load_model_from_path,
            quantization_config=quantization_config,
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{model_name}",
            quantization_config=quantization_config,
        )

    if resume_from_checkpoint and peft_method is None:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model = model.from_pretrained(
            resume_from_checkpoint,
            quantization_config=quantization_config,
        )

    if dialect is not None:
        model.resize_token_embeddings(len(processor.tokenizer))
        # Save the model with the new embeddings
        model.save_pretrained(output_dir / "base_model")

    if peft_method:
        if peft_method == "lora" or peft_method == "qlora":
            _peft_config = _args.get("LoraConfig", {})
            peft_config = LoraConfig(
                inference_mode=False,
                **_peft_config
            )
            logging.info(f"PEFT Config: {peft_config}")
            if peft_method == "qlora":
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing)
                model.config.use_reentrant = True
            else:
                model.config.use_reentrant = False
            model = get_peft_model(model, peft_config)
            if dialect is not None:
                # Modify the peft config so that it points to the new base model
                model.peft_config['default'].base_model_name_or_path = str(
                    output_dir.absolute() / "base_model")
            model.print_trainable_parameters()
            training_args.output_dir = str(output_dir)
        else:
            raise ValueError(f"Unknown PEFT method: {peft_method}")

    if src_lang != "all":
        _language = LANGS[src_lang]
        if dialect is not None:
            _language = DIALECT[dialect]['dialect']
            # Update the lang_to_id mapping in the model's generation config
            # i.e. adding an entry in the dictionary in the format of '<|dialect|>': <dialect_id>
            # Note that the id is fetched from the tokenizers' vocab
            dialect_token = f'<|{TO_LANGUAGE_CODE[DIALECT[dialect]["dialect"]]}|>'
            dialect_token_id = processor.tokenizer.convert_tokens_to_ids(
                dialect_token)
            model.generation_config.lang_to_id.update(
                {dialect_token: dialect_token_id})
        model.generate = partial(
            model.generate, language=_language, task="transcribe" if mode == "asr" else "translate")

    # Apply SpecAugment if specified
    if _args.get("WhisperConfig", None) is not None and _args["WhisperConfig"].get("apply_spec_augment", False):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = _args['WhisperConfig'].get(
            "mask_time_prob", 0.05)
        model.config.mask_time_length = _args['WhisperConfig'].get(
            "mask_time_length", 10)
        model.config.mask_time_min_masks = _args['WhisperConfig'].get(
            "mask_time_min_masks", 2)
        model.config.mask_feature_prob = _args['WhisperConfig'].get(
            "mask_feature_prob", 0.05)
        model.config.mask_feature_length = _args['WhisperConfig'].get(
            "mask_feature_length", 10)
        model.config.mask_feature_min_masks = _args['WhisperConfig'].get(
            "mask_feature_min_masks", 0)

    class PrinterCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_local_process_zero:
                _alpha = _schedule_dynamic_mtl_weight(
                    current_step=state.global_step,
                    max_steps=state.max_steps,
                    warmup_steps=self._trainer.loss_warmup if loss_warmup != -1 else args.warmup_steps,
                    loss_base=self._trainer.loss_base,
                    min_weight=self._trainer.min_alpha,
                    max_weight=self._trainer.max_alpha,
                )
                logs["st_loss_weight"] = _alpha
                print(logs)

    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=_train_dset,
        eval_dataset=_val_dset,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        use_asr_prompt=use_asr_prompt,
        src_lang=LANGS[src_lang],
        eval_steps=10,
        min_promptless_prob=min_promptless_prob,
        max_promptless_prob=max_promptless_prob,
        max_sample_prob=max_sample_prob,
        min_sample_prob=min_sample_prob,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        loss_warmup=loss_warmup,
        loss_base=loss_base,
    )
    trainer.add_callback(PrinterCallback(trainer))

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
                        choices=["asr", "st", "mtl", "pmtl"],
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
                        choices=["lora", "qlora", None],
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
    parser.add_argument("--dialect", type=str, default=None,
                        help="The dialect language code will be used instead of the src_lang code for training if specified.")
    parser.add_argument("--use-asr-prompt", action="store_true",
                        help="Whether to use the ASR hyp/ref as the prompt for the ST task.")
    parser.add_argument("--min-promptless-prob", type=float, default=0.0,
                        help="The minimum probability for performing promptless ST finetuning.")
    parser.add_argument("--max-promptless-prob", type=float, default=0.0,
                        help="The minimum probability for performing promptless ST finetuning.")
    parser.add_argument("--min-sample-prob", type=float, default=0.0,
                        help="Minimum sampling probability for the BMTL training.")
    parser.add_argument("--max-sample-prob", type=float, default=0.0,
                        help="Maximum sampling probability for the BMTL training. \
                        Note that the sampling mechanism is disabled if this is set to 0.0.\
                        Also note that the sampling probability will be linearly increased from min_sample_prob to max_sample_prob during the training AFTER the warmup steps.")
    parser.add_argument("--min-alpha", type=float, default=.5,
                        help="Minimum alpha for the PMTL training.")
    parser.add_argument("--max-alpha", type=float, default=.5,
                        help="Maximum alpha for the PMTL training. \
                        Note that the ST loss weight will be constant if min_alpha == max_alpha.\
                        Also note that the alpha will increase at logarithmically from min_alpha to max_alpha \
                        during the training AFTER the loss_warmup steps.")
    parser.add_argument("--loss-warmup", type=int, default=-1,
                        help="Number of steps to warm up the loss weight for the PMTL training.")
    parser.add_argument("--loss-base", type=float, default=.25,
                        help="Base for the log-increase ST weight, larger means slower increases.")

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
            dialect=args.dialect,
            use_asr_prompt=args.use_asr_prompt,
            min_promptless_prob=args.min_promptless_prob,
            max_promptless_prob=args.max_promptless_prob,
            max_sample_prob=args.max_sample_prob,
            min_sample_prob=args.min_sample_prob,
            max_alpha=args.max_alpha,
            min_alpha=args.min_alpha,
            loss_warmup=args.loss_warmup,
            loss_base=args.loss_base,
        )


if __name__ == "__main__":
    main()
