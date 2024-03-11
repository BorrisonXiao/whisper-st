#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Change the following according to your experiments
# src_lang=kor
# src_lang=ara
# src_lang=cmn
src_lang=spa
# src_lang=rus
# src_lang=all
tgt_lang=eng

# Use the dialectal prefix
# dialect=tus
dialect=

train_set=train-cts
# train_set=train-all
train_dev=dev1
extra_dev=dev2

# debug=true
debug=false

ds_config=conf/tuning/ds2.json # The deepspeed configuration file
merge_utt=true                 # Whether to merge utterances for training. This is particularly important for finetuning.
remove_ark=true                # Whether to remove the ark files generated by kaldi to save disk space
peft_method=lora               # none, lora, qlora
prompted_mtl=true              # Whether to use the prompted multi-task learning
normalize_text=false           # Whether or not to normalize the text at training time
master_port=29501              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=8                 # Number of jobs for decoding, note that each job will use a GPU
use_gpu_inference=true         # Whether to use GPU for inference
merge_decode=false             # Whether to merge the utterances at decoding time
skip_data_prep=true            # Whether to skip data preparation
skip_training=false            # Whether to skip training
use_asr_prompt=true            # Whether to mask the ASR hypothesis at BMTL training time
min_promptless_prob=0.2        # The minimum probability for performing promptless ST finetuning
max_promptless_prob=0.2        # The maximum probability for perforFming promptless ST finetuning
min_sample_prob=0.6            # The minimum probability for sampling the PMTL ASR hypothesis
max_sample_prob=0.8            # The maximum probability for sampling the PMTL ASR hypothesis (0.0 means disable sampling)
min_alpha=0.4                  # The minimum alpha for the multi-task losses, i.e. the weight for the ST loss
max_alpha=0.5                  # The maximum alpha for the multi-task losses, i.e. the weight for the ST loss (0.0 means disable ST loss)
dynamic_loss_start_step=1      # The step to start the dynamic loss weight
dynamic_loss_k=0.25            # The k for the dynamic loss weight (the log base)
use_asr_prompt_decode=false    # Whether to mask the ASR hypothesis at inference time
promptless_decode=false        # Whether to perform promptless decoding at inference time
disable_asr_inference=true     # Whether to disable ASR inference at inference time, note this only works when use_asr_prompt_decode is false
load_model_from_path=          # The path to load the model from
resume_from_checkpoint=        # The path to resume from a checkpoint

# Modify this to your python path, this is due to some ESPNet environment issues
python_hf=python3
# The database for storing merged data
merged_data_base=

opts=
data_opts=
if "${debug}"; then
    model=large-v2 # base, large, large-v2 etc.
    st_config=conf/tuning/whisper-debug.yaml
    mtl_config=conf/tuning/whisper-debug.yaml
    resume_from_checkpoint=
else
    model=large-v2 # base, large, large-v2 etc.
    st_config=conf/tuning/st_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    if "${prompted_mtl}"; then
        mtl_config=conf/tuning/mtl_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    else
        mtl_config=conf/tuning/mtl_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    fi
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
fi

if [ ${model} == "large-v2" ]; then
    inference_batch_size=24
elif [ ${model} == "medium" ]; then
    inference_batch_size=48
elif [ ${model} == "tiny" ]; then
    inference_batch_size=128
fi

if "${merge_utt}"; then
    _suf="_merged"
else
    _suf=
fi
# Where to save the output at evaluation time
_lang=${src_lang}
if [ -n "${dialect}" ]; then
    _lang=${dialect}
fi
asr_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}${_suf}/${_lang}/${train_set}_sp/asr/${peft_method}/logdir/eval_preds.txt
if "${prompted_mtl}"; then
    st_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}${_suf}/${_lang}/${train_set}_sp/pmtl/${peft_method}/logdir/eval_preds.txt
else
    st_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}${_suf}/${_lang}/${train_set}_sp/st/${peft_method}/logdir/eval_preds.txt
fi

if [ -n "${load_model_from_path}" ]; then
    opts+=" --load_model_from_path ${load_model_from_path} "
fi
if [ -n "${resume_from_checkpoint}" ]; then
    opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
fi
opts+=" --debug ${debug} "
if [ -n "${asr_save_eval_preds}" ]; then
    opts+=" --asr_save_eval_preds ${asr_save_eval_preds} "
fi
if [ -n "${st_save_eval_preds}" ]; then
    opts+=" --st_save_eval_preds ${st_save_eval_preds} "
fi
opts+=" --use_asr_prompt ${use_asr_prompt}"
opts+=" --min_promptless_prob ${min_promptless_prob} "
opts+=" --max_promptless_prob ${max_promptless_prob} "
opts+=" --min_sample_prob ${min_sample_prob} "
opts+=" --max_sample_prob ${max_sample_prob} "
opts+=" --min_alpha ${min_alpha} "
opts+=" --max_alpha ${max_alpha} "
opts+=" --dynamic_loss_start_step ${dynamic_loss_start_step} "
opts+=" --dynamic_loss_k ${dynamic_loss_k} "

declare -A testset_dict

testset_dict+=(
    ["ara"]="iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test"
    ["kor"]="uhura_test"
    ["rus"]="uhura_test"
    ["spa"]="fisher_test"
    ["all"]="iwslt22_test bbn_cts_bolt_test uhura_test fisher_test callhome_test")

test_set=${testset_dict[${src_lang}]} # This option is to run eval
# test_set="fleurs_test"

framework=huggingface # huggingface, openai
preprocessing_num_proc=16
on_the_fly_feat=false

src_case=tc #lc.rm
tgt_case=tc #lc.rm

# Data prep extra options
stereo=true
ignore_segments=false
fs_str=16000
fs=16k
min_duration=0.0
start_at_zero=true
if "${merge_utt}"; then
    hf_datadir=
    datadir=data/${src_lang}
    dumpdir=dump_gaussian/${src_lang}
    opts+=' --merged_data_base '
    opts+=$merged_data_base
    data_opts+=' --merged_data_base '
    data_opts+=$merged_data_base
else
    hf_datadir=
    datadir=data/${src_lang}
    dumpdir=dump_gaussian/${src_lang}
fi
if [ -n "$dialect" ]; then
    opts+=' --dialect '
    opts+=$dialect
fi

# There might be a better way to do this, maybe passing a yaml file that gets parsed by the local/data.sh
local_data_opts='--stage 0 --stop_stage 100 --fs_str '
local_data_opts+=$fs_str
local_data_opts+=' --stereo '
local_data_opts+=$stereo
local_data_opts+=' --ignore_segments '
local_data_opts+=$ignore_segments
local_data_opts+=' --min_duration '
local_data_opts+=$min_duration
local_data_opts+=' --start_at_zero '
local_data_opts+=$start_at_zero
local_data_opts+=' --train_set '
local_data_opts+=$train_set
local_data_opts+=' --dev_set '
local_data_opts+=$train_dev
local_data_opts+=' --src_lang '
local_data_opts+=$src_lang
local_data_opts+=' --datadir '
local_data_opts+=$datadir

if ! "${skip_data_prep}"; then
    ./data.sh \
        --local_data_opts "$local_data_opts" \
        --audio_format "flac.ark" \
        --nj 80 \
        --fs ${fs} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --src_case ${src_case} \
        --tgt_case ${tgt_case} \
        --feats_type raw \
        --speed_perturb_factors "0.9 1.0 1.1" \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 6 \
        --stop_stage 6 \
        --datadir ${datadir} \
        --dumpdir "${dumpdir}" \
        --save_wav true \
        --framework ${framework} \
        --hf_datadir ${hf_datadir} \
        --extra_valid_set "${extra_dev}" \
        --merge_utt ${merge_utt} \
        --remove_ark ${remove_ark} \
        --gaussian_merge ${prompted_mtl} \
        --python_hf ${python_hf} ${data_opts}
fi

if ! "${skip_training}"; then
    ./prompted_ft.sh \
        --ngpu 8 \
        --expdir ft_exp \
        --local_data_opts "$local_data_opts" \
        --nj 80 \
        --st_config ${st_config} \
        --mtl_config ${mtl_config} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --feats_type raw \
        --speed_perturb_factors "0.9 1.0 1.1" \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 2 \
        --stop_stage 3 \
        --dumpdir "${dumpdir}" \
        --st_tag whisper_${model} \
        --model_name ${model} \
        --use_gpu_inference ${use_gpu_inference} \
        --inference_nj ${inference_nj} \
        --framework ${framework} \
        --hf_datadir ${hf_datadir} \
        --peft_method ${peft_method} \
        --preprocessing_num_proc ${preprocessing_num_proc} \
        --on_the_fly_feat ${on_the_fly_feat} \
        --dev_name ${train_dev} \
        --extra_valid_set "${extra_dev}" \
        --merge_utt ${merge_utt} \
        --normalize_text ${normalize_text} \
        --master_port ${master_port} \
        --python_hf ${python_hf} \
        --inference_batch_size ${inference_batch_size} \
        --use_asr_prompt_decode ${use_asr_prompt_decode} \
        --promptless_decode ${promptless_decode} \
        --disable_asr_inference ${disable_asr_inference} \
        --num_beams 1 \
        --merge_decode ${merge_decode} ${opts}
fi
