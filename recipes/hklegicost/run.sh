#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Change the following according to your experiments
train_set=train
train_dev=dev-asr-0
extra_dev=dev-mt-0

# debug=true
debug=false

ds_config=conf/tuning/ds2.json # The deepspeed configuration file
peft_method=lora               # none, lora, qlora
normalize_text=false           # Whether or not to normalize the text at training time
master_port=29500              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=4                 # Number of jobs for decoding, note that each job will use a GPU
skip_data_prep=true            # Whether to skip data preparation
python_hf=python3

opts=
if "${debug}"; then
    model=tiny # base, large, large-v2 etc.
    asr_config=conf/tuning/whisper-debug.yaml
    st_config=conf/tuning/whisper-debug.yaml
    mtl_config=conf/tuning/whisper-debug.yaml
    resume_from_checkpoint=
else
    model=tiny # base, large, large-v2 etc.
    asr_config=conf/tuning/asr_${model}_${peft_method}_${train_set}.yaml
    st_config=conf/tuning/st_${model}_${peft_method}_${train_set}.yaml
    mtl_config=conf/tuning/mtl_${model}_${peft_method}_${train_set}.yaml
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
    resume_from_checkpoint=
fi

if [ ${model} == "large-v2" ]; then
    inference_batch_size=12
elif [ ${model} == "medium" ]; then
    inference_batch_size=24
elif [ ${model} == "tiny" ]; then
    inference_batch_size=64
fi

asr_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}/cmn/${train_set}/asr/${peft_method}/logdir/eval_preds.txt
st_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}/cmn/${train_set}/st/${peft_method}/logdir/eval_preds.txt

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

test_set="test" # This option is to run eval

framework=huggingface # huggingface, openai
preprocessing_num_proc=16
on_the_fly_feat=false

# Data prep extra options
stereo=true
ignore_segments=false
fs_str=16000
min_duration=0.0
dumpdir=dump
datadir=${dumpdir}/raw
src_lang="cmn"

# There might be a better way to do this, maybe passing a yaml file that gets parsed by the local/data.sh
local_data_opts='--stage 0 --stop_stage 100 --fs_str '
local_data_opts+=$fs_str
local_data_opts+=' --stereo '
local_data_opts+=$stereo
local_data_opts+=' --ignore_segments '
local_data_opts+=$ignore_segments
local_data_opts+=' --min_duration '
local_data_opts+=$min_duration
local_data_opts+=' --train_set '
local_data_opts+=$train_set
local_data_opts+=' --dev_set '
local_data_opts+=$train_dev
local_data_opts+=' --src_lang '
local_data_opts+=$src_lang
local_data_opts+=' --datadir '
local_data_opts+=$datadir

if ! "${skip_data_prep}"; then
    ./data.sh
fi

./finetune.sh \
    --ngpu 1 \
    --expdir ft_exp \
    --local_data_opts "$local_data_opts" \
    --nj 80 \
    --st_config ${st_config} \
    --asr_config ${asr_config} \
    --mtl_config ${mtl_config} \
    --src_lang ${src_lang} \
    --tgt_lang ${src_lang} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --stage 1 \
    --stop_stage 1 \
    --dumpdir "${dumpdir}" \
    --st_tag whisper_${model} \
    --model_name ${model} \
    --inference_nj ${inference_nj} \
    --framework ${framework} \
    --hf_datadir ${datadir} \
    --peft_method ${peft_method} \
    --preprocessing_num_proc ${preprocessing_num_proc} \
    --on_the_fly_feat ${on_the_fly_feat} \
    --dev_name ${train_dev} \
    --extra_valid_set "${extra_dev}" \
    --normalize_text ${normalize_text} \
    --master_port ${master_port} \
    --python_hf ${python_hf} \
    --inference_batch_size ${inference_batch_size} \
    ${opts}
