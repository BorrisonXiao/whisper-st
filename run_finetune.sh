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
src_lang=cmn
# src_lang=spa
# src_lang=rus
# src_lang=all
tgt_lang=eng

train_set=train-cts
train_dev=dev1
# debug=true
debug=false
ds_config=conf/tuning/ds2.json

opts=
if "${debug}"; then
    st_config=conf/tuning/whisper-debug.yaml
    model=tiny # base, large, large-v2 etc.
    resume_from_checkpoint=
else
    st_config=conf/tuning/finetune_asr_whisper_large-v2_cmn.yaml
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
    model=large-v2 # base, large, large-v2 etc.
    # resume_from_checkpoint=ft_exp/hf_whisper_large-v2/cmn/asr/lora/checkpoint-14000
    resume_from_checkpoint=
fi
# ds_config=conf/tuning/ds3.json
save_eval_preds=/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2/cmn/asr/lora/logdir/eval_preds.txt

if [ -n "${resume_from_checkpoint}" ]; then
    opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
fi
opts+=" --debug ${debug} "
if [ -n "${save_eval_preds}" ]; then
    opts+=" --save_eval_preds ${save_eval_preds} "
fi

declare -A testset_dict

testset_dict+=(
    ["ara"]="iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test"
    ["kor"]="uhura_test"
    ["rus"]="uhura_test"
    ["spa"]="fisher_test callhome_test")

# test_set=${testset_dict[${src_lang}]} # This option is to run eval
test_set="fleurs_test"

framework=huggingface # huggingface, openai
inference_nj=4 # Number of jobs for decoding, note that each job will use a GPU
mode=asr # asr, st, mtl
# framework=openai # huggingface, openai
peft_method=lora # none, lora, qlora
preprocessing_num_proc=16
on_the_fly_feat=true

src_case=tc #lc.rm
tgt_case=tc #lc.rm

# Data prep extra options
stereo=true
ignore_segments=false
fs_str=16000
fs=16k
min_duration=0.0
start_at_zero=true
datadir=data/${src_lang}
# hf_datadir=/expscratch/dchakraborty/hf_datasets/scale23/data/all
hf_datadir=/exp/cxiao/scale23/hf_data

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
    --ngpu 4 \
    --expdir ft_exp \
    --local_data_opts "$local_data_opts" \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_joint false \
    --nj 80 \
    --fs ${fs} \
    --st_config ${st_config} \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
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
    --stage 6 \
    --stop_stage 6 \
    --datadir ${datadir} \
    --dumpdir "dump/${src_lang}" \
    --save_wav true \
    --st_tag whisper_${model} \
    --model_name ${model} \
    --mode ${mode} \
    --inference_nj ${inference_nj} \
    --framework ${framework} \
    --hf_datadir ${hf_datadir} \
    --peft_method ${peft_method} \
    --preprocessing_num_proc ${preprocessing_num_proc} \
    --on_the_fly_feat ${on_the_fly_feat} \
    --dev_name ${train_dev} \
    --skip_data_prep true ${opts}