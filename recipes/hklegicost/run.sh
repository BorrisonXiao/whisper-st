#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Change the following according to your experiments
src_lang=cmn
tgt_lang=eng

# Use the dialectal prefix
# dialect=tus
dialect=

# Change the following according to your experiments
train_set=train
train_dev=dev-asr-0

# debug=true
debug=false

ds_config=conf/tuning/ds2.json                                              # The deepspeed configuration file
peft_method=lora                                                            # none, lora, qlora
prompted_mtl=false                                                          # Whether to use the prompted multi-task learning
normalize_text=false                                                        # Whether or not to normalize the text at training time
master_port=29501                                                           # Master port for distributed training (to avoid conflict on the same node)
inference_nj=2                                                              # Number of jobs for decoding, note that each job will use a GPU
use_gpu_inference=true                                                      # Whether to use GPU for inference
skip_data_prep=true                                                         # Whether to skip data preparation
skip_training=false                                                         # Whether to skip training
load_model_from_path=                                                       # The path to load the model from
stm_dir=                                                                    # The directory where the raw data is stored (in stm format)
resume_from_checkpoint=                                                     # The path to resume from a checkpoint
num_beams=2                                                                 # Number of beams for decoding
eval_cer=false                                                              # Whether to evaluate CER at evaluation time (if false, WER will be used)
pseudo_st=true                                                              # Whether to use pseudo-translation for training
dumpdir=dump                                                                # The directory where the dump is stored
hf_datadir=/home/cxiao7/research/whisper-st/recipes/hklegicost/dump/hf_data # The directory where the Huggingface style data is stored

# Modify this to your python path, this is due to some ESPNet environment issues
python_hf=python3

datadir=data/${src_lang}
opts=
data_opts=

if "${debug}"; then
    model=large-v2 # base, large, large-v2 etc.
    asr_config=conf/tuning/whisper-debug.yaml
    st_config=conf/tuning/whisper-debug.yaml
    mtl_config=conf/tuning/whisper-debug.yaml
    resume_from_checkpoint=
    stm_dir=/home/cxiao7/research/whisper-st/recipes/hklegicost/data/cmn/resegmented
else
    model=large-v2 # base, large, large-v2 etc.
    asr_config=conf/tuning/asr_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    st_config=conf/tuning/st_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    if "${prompted_mtl}"; then
        mtl_config=conf/tuning/mtl_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    else
        mtl_config=conf/tuning/mtl_${model}_${src_lang}_${peft_method}_${train_set}.yaml
    fi
    if [ -n "${ds_config}" ]; then
        opts+=" --ds_config ${ds_config} "
    fi
    stm_dir=/home/cxiao7/research/whisper-st/recipes/hklegicost/data/cmn/resegmented
fi

if [ ${model} == "large-v2" ]; then
    inference_batch_size=48
elif [ ${model} == "medium" ]; then
    inference_batch_size=48
elif [ ${model} == "tiny" ]; then
    inference_batch_size=128
fi

# Adjust the batch size based on the number of beams
inference_batch_size=$((inference_batch_size / num_beams))

_lang=${src_lang}
if [ -n "${dialect}" ]; then
    _lang=${dialect}
fi

asr_save_eval_preds=${PWD}/exp/hf_whisper_${model}/cmn/${train_set}/asr/${peft_method}/logdir/eval_preds.txt
st_save_eval_preds=${PWD}/exp/hf_whisper_${model}/cmn/${train_set}/st/${peft_method}/logdir/eval_preds.txt

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
if [ -n "${stm_dir}" ]; then
    opts+=" --stm_dir ${stm_dir} "
fi

test_set="test" # This option is to run eval

framework=huggingface # huggingface, openai
preprocessing_num_proc=40
on_the_fly_feat=false

if ! "${skip_data_prep}"; then
    ./data.sh \
        --nj 80 \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --feats_type raw \
        --speed_perturb_factors "0.9 1.0 1.1" \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 2 \
        --stop_stage 2 \
        --datadir ${datadir} \
        --dumpdir ${PWD}/${dumpdir} \
        --hf_datadir ${hf_datadir} \
        --srcdir /home/cxiao7/research/legicost/export \
        --python_hf ${python_hf} ${data_opts}
fi

if ! "${skip_training}"; then
    ./finetune.sh \
        --ngpu 2 \
        --expdir exp \
        --nj 80 \
        --asr_config ${asr_config} \
        --st_config ${st_config} \
        --mtl_config ${mtl_config} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --feats_type raw \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 6 \
        --stop_stage 6 \
        --dumpdir "${dumpdir}" \
        --st_tag whisper_${model} \
        --model_name ${model} \
        --gpu_inference ${use_gpu_inference} \
        --inference_nj ${inference_nj} \
        --framework ${framework} \
        --hf_datadir ${hf_datadir} \
        --peft_method ${peft_method} \
        --preprocessing_num_proc ${preprocessing_num_proc} \
        --on_the_fly_feat ${on_the_fly_feat} \
        --dev_name ${train_dev} \
        --normalize_text ${normalize_text} \
        --master_port ${master_port} \
        --python_hf ${python_hf} \
        --inference_batch_size ${inference_batch_size} \
        --eval_cer ${eval_cer} \
        --pseudo_st ${pseudo_st} \
        --num_beams ${num_beams} ${opts}
fi
