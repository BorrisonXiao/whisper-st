#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Change the following according to your experiments
src_lang=kor
# src_lang=ara
# src_lang=ara
# src_lang=spa
# src_lang=rus
# src_lang=all
tgt_lang=eng

# train_set=train-cts
train_set=train-all
train_dev=dev1
extra_dev=dev2

# debug=true
debug=false

ds_config=conf/tuning/ds2.json # The deepspeed configuration file
merge_utt=true                 # Whether to merge utterances for training. This is particularly important for finetuning.
remove_ark=true                # Whether to remove the ark files generated by kaldi to save disk space
peft_method=lora               # none, lora, qlora
normalize_text=false           # Whether or not to normalize the text at training time
master_port=29501              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=7                 # Number of jobs for decoding, note that each job will use a GPU

# Modify this to your python path, this is due to some ESPNet environment issues
python_hf=/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3

# The database for storing merged data
merged_data_base=/exp/cxiao/scale23/merged_data_base

opts=
if "${debug}"; then
    model=tiny # base, large, large-v2 etc.
    asr_config=conf/tuning/whisper-debug.yaml
    st_config=conf/tuning/whisper-debug.yaml
    resume_from_checkpoint=
else
    model=large-v2 # base, large, large-v2 etc.
    asr_config=conf/tuning/asr_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    # For now, using the same config for ST
    st_config=conf/tuning/st_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
    resume_from_checkpoint=
fi

if "${merge_utt}"; then
    _suf="_merged"
else
    _suf=
fi
# Where to save the output at evaluation time
asr_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}${_suf}/${src_lang}/${train_set}_sp/asr/${peft_method}/logdir/eval_preds.txt
st_save_eval_preds=${PWD}/ft_exp/hf_whisper_${model}${_suf}/${src_lang}/${train_set}_sp/st/${peft_method}/logdir/eval_preds.txt

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

declare -A testset_dict

testset_dict+=(
    ["ara"]="iwslt22_test fleurs_test"
    ["cmn"]="bbn_cts_bolt_test fleurs_test"
    ["kor"]="uhura_test fleurs_test"
    ["rus"]="uhura_test fleurs_test"
    ["spa"]="fisher_test callhome_test fleurs_test")

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
    hf_datadir=/exp/cxiao/scale23/_merged_hf_data
    datadir=data/${src_lang}
    dumpdir=dump/${src_lang}
    opts+=' --merged_data_base '
    opts+=$merged_data_base
else
    hf_datadir=/exp/cxiao/scale23/hf_data
    datadir=data/${src_lang}
    dumpdir=dump/${src_lang}
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

./finetune.sh \
    --ngpu 8 \
    --expdir ft_exp \
    --local_data_opts "$local_data_opts" \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_joint false \
    --nj 80 \
    --fs ${fs} \
    --st_config ${st_config} \
    --asr_config ${asr_config} \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --tgt_token_type "bpe" \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --stage 10 \
    --stop_stage 12 \
    --datadir ${datadir} \
    --dumpdir "${dumpdir}" \
    --save_wav true \
    --st_tag whisper_${model} \
    --model_name ${model} \
    --inference_nj ${inference_nj} \
    --framework ${framework} \
    --hf_datadir ${hf_datadir} \
    --peft_method ${peft_method} \
    --preprocessing_num_proc ${preprocessing_num_proc} \
    --on_the_fly_feat ${on_the_fly_feat} \
    --dev_name ${train_dev} \
    --extra_valid_set "${extra_dev}" \
    --merge_utt ${merge_utt} \
    --remove_ark ${remove_ark} \
    --normalize_text ${normalize_text} \
    --master_port ${master_port} \
    --python_hf ${python_hf} \
    --skip_data_prep false ${opts}
