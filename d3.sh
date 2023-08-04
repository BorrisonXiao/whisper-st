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
# src_lang=cmn
# src_lang=spa
# src_lang=rus
# src_lang=all
tgt_lang=eng

train_set=train-cts
train_dev=dev1
extra_dev=dev2

# debug=true
debug=false

merge_utt=false                # Whether to merge utterances for training. This is particularly important for finetuning.
remove_ark=true                # Whether to remove the ark files generated by kaldi to save disk space
normalize_text=false           # Whether or not to normalize the text at training time
inference_nj=8                 # Number of jobs for decoding, note that each job will use a GPU
merge_decode=false             # Whether to merge the utterances at decoding time

# Modify this to your python path, this is due to some ESPNet environment issues
python_hf=/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3

# The database for storing merged data
merged_data_base=/exp/cxiao/scale23/merged_data_base

opts=
model=medium # base, large, large-v2 etc.

if [ ${model} == "large-v2" ]; then
    inference_batch_size=32
elif [ ${model} == "medium" ]; then
    inference_batch_size=48
fi
opts+=" --debug ${debug} "

declare -A testset_dict

testset_dict+=(
    ["ara"]="iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test"
    ["kor"]="uhura_test"
    ["rus"]="uhura_test"
    ["spa"]="fisher_test callhome_test")

test_set=${testset_dict[${src_lang}]} # This option is to run eval
# test_set="fleurs_test"

framework=huggingface # huggingface, openai

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

./decode2.sh \
    --ngpu 8 \
    --expdir decode_exp \
    --local_data_opts "$local_data_opts" \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_joint false \
    --nj 80 \
    --fs ${fs} \
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
    --stage 8 \
    --stop_stage 12 \
    --datadir ${datadir} \
    --dumpdir "${dumpdir}" \
    --save_wav true \
    --st_tag whisper_${model} \
    --model_name ${model} \
    --inference_nj ${inference_nj} \
    --framework ${framework} \
    --hf_datadir ${hf_datadir} \
    --extra_valid_set "${extra_dev}" \
    --remove_ark ${remove_ark} \
    --python_hf ${python_hf} \
    --inference_batch_size ${inference_batch_size} \
    --merge_decode ${merge_decode} \
    --skip_data_prep false ${opts}
