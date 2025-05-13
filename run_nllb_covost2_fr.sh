#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Change the following according to your experiments
src_lang=fr
tgt_lang=en

train_set=train
train_dev=validation
extra_dev=validation

debug=false
# debug=true

ds_config=conf/tuning/ds2.json # The deepspeed configuration file
peft_method=none               # none, lora, qlora
master_port=29506              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=4                 # Number of jobs for decoding, note that each job will use a GPU
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

declare -A testset_dict

testset_dict+=(
    ["fr"]="test")

test_set=${testset_dict[${src_lang}]} # This option is to run eval

hf_datadir=/exp/cxiao/scale23/nllb_data_covost2/${src_lang}
datadir=data/${src_lang}
dumpdir=dump_covost2/${src_lang}

if ! "${skip_training}"; then
    ./nllb_covost2.sh \
        --ngpu 4 \
        --expdir exp_nllb_covost2 \
        --mt_config ${mt_config} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 3 \
        --stop_stage 3 \
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
