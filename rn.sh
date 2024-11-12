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

debug=false
# debug=true

ds_config=conf/tuning/ds2.json # The deepspeed configuration file
peft_method=none               # none, lora, qlora
master_port=29501              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=8                 # Number of jobs for decoding, note that each job will use a GPU
skip_data_prep=true            # Whether to skip data preparation
skip_training=false            # Whether to skip training
load_model_from_path=          # The path to load the model from
resume_from_checkpoint=        # The path to resume from a checkpoint
preprocessing_num_proc=8       # Number of processes for preprocessing

# Modify this to your python path, this is due to some ESPNet environment issues
python=python3

opts=
if "${debug}"; then
    model=1.3B
    mt_config=conf/tuning/whisper-debug.yaml
    resume_from_checkpoint=
else
    model=1.3B
    mt_config=conf/tuning/mt_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
fi

if [ ${model} == "1.3B" ]; then
    inference_batch_size=10
fi

if [ -n "${load_model_from_path}" ]; then
    opts+=" --load_model_from_path ${load_model_from_path} "
fi
if [ -n "${resume_from_checkpoint}" ]; then
    opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
fi
opts+=" --debug ${debug} "
opts+=" --merged_data_base /home/hltcoe/cxiao/scale23/whisper/recipe/st/gaussian_data_base "
opts+=" --merged_data_dir /exp/cxiao/scale23/_gaussian_hf_data "

declare -A testset_dict

testset_dict+=(
    ["ara"]="iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test"
    ["kor"]="uhura_test"
    ["rus"]="uhura_test"
    ["spa"]="fisher_test"
    ["all"]="iwslt22_test bbn_cts_bolt_test uhura_test fisher_test callhome_test")

test_set=${testset_dict[${src_lang}]} # This option is to run eval

hf_datadir=/exp/cxiao/scale23/hf_data
datadir=data/${src_lang}
dumpdir=dump/${src_lang}

if ! "${skip_training}"; then
    ./nllb.sh \
        --ngpu 8 \
        --expdir exp_nllb \
        --nj 80 \
        --mt_config ${mt_config} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 7 \
        --stop_stage 8 \
        --datadir "${datadir}" \
        --dumpdir "${dumpdir}" \
        --model_name ${model} \
        --inference_nj ${inference_nj} \
        --hf_datadir ${hf_datadir} \
        --peft_method ${peft_method} \
        --extra_valid_set "${extra_dev}" \
        --master_port ${master_port} \
        --preprocessing_num_proc ${preprocessing_num_proc} \
        --inference_batch_size ${inference_batch_size} \
        --num_beams 1 ${opts}
fi
