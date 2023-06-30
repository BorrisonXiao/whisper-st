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
tgt_lang=eng

train_set=train-cts
train_dev=dev

declare -A testset_dict

testset_dict+=(
    ["ara"]="fleurs_test iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test fleurs_test"
    ["kor"]="uhura_test fleurs_test"
    ["rus"]="uhura_test fleurs_test"
    ["spa"]="fisher_test callhome_test fleurs_test")

# test_set=${testset_dict[${src_lang}]} # This option is to run eval
test_set="fleurs_test"

#st_config=conf/train_st.yaml
st_config=conf/train_st_baseline.yaml
inference_config=conf/decode_st.yaml

framework=huggingface # huggingface, openai
# framework=openai # huggingface, openai
model=large-v2 # base, large, large-v2 etc.
# model=base # base, large, large-v2 etc.
inference_nj=8 # Number of jobs for decoding, note that each job will use a GPU

src_nbpe=2000
tgt_nbpe=2000

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
hf_datadir=/expscratch/dchakraborty/hf_datasets/scale23/data/all

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

./decode.sh \
    --use_streaming false \
    --local_data_opts "$local_data_opts" \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_joint false \
    --nj 80 \
    --fs ${fs} \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@" \
    --stage 6 \
    --stop_stage 9 \
    --datadir ${datadir} \
    --dumpdir "dump/${src_lang}" \
    --save_wav true \
    --st_tag whisper_${model} \
    --model_name ${model} \
    --inference_nj ${inference_nj} \
    --framework ${framework} \
    --hf_datadir ${hf_datadir}
