#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# This script is used to run the final mult-task, i.e. ST + ASR + MT + Prompted-ST, experiments
# Note that this version uses masked GT prompt at training time

# Change the following according to your experiments
src_lang=fr
tgt_lang=en

train_set=train-cts
mt_train_set=train-cts
train_dev=validation
extra_dev=validation2

debug=false
# debug=true

ds_config=conf/tuning/ds2.json # The deepspeed configuration file
merge_utt=true                 # Whether to merge utterances for training. This is particularly important for finetuning.
peft_method=lora               # none, lora, qlora
prompted_mtl=true              # Whether to use the prompted multi-task learning
normalize_text=false           # Whether or not to normalize the text at training time
master_port=29501              # Master port for distributed training (to avoid conflict on the same node)
inference_nj=8                 # Number of jobs for decoding, note that each job will use a GPU
use_gpu_inference=true         # Whether to use GPU for inference
skip_data_prep=false           # Whether to skip data preparation
skip_training=true             # Whether to skip training
use_asr_prompt=true            # Whether to mask the ASR hypothesis at BMTL training time
min_promptless_prob=0.2        # The minimum probability for performing promptless ST finetuning
max_promptless_prob=0.2        # The maximum probability for perforFming promptless ST finetuning
batch_mask_prob=0.8            # The probability for applying masks to the prompt
token_mask_prob=0.5            # The probability for masking tokens in the prompt
min_alpha=0.4                  # The minimum alpha for the multi-task losses, i.e. the weight for the ST loss
max_alpha=0.5                  # The maximum alpha for the multi-task losses, i.e. the weight for the ST loss (0.0 means disable ST loss)
dynamic_loss_start_step=1      # The step to start the dynamic loss weight
dynamic_loss_k=0.25            # The k for the dynamic loss weight (the log base)
use_asr_prompt_decode=true     # Whether to use the ASR hypothesis at inference time
promptless_decode=false        # Whether to perform promptless decoding at inference time
disable_asr_inference=true     # Whether to disable ASR inference at inference time, note this only works when use_asr_prompt_decode is false
use_asr_prompt_dev=false       # Whether to use ASR prompt at dev time
load_model_from_path=          # The path to load the model from
resume_from_checkpoint=        # The path to resume from a checkpoint
eval_multi_bleu=false          # Whether to evaluate the multi-BLEU score (fisher-spanish only) for the MT task
no_glm=true                    # Whether to use the GLM for evaluation

# Modify this to your python path, this is due to some ESPNet environment issues
python_hf=python3

min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}

opts=
data_opts=
if "${debug}"; then
    # model=tiny # base, large, large-v2 etc.
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
    inference_batch_size=16
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

if [ -n "${load_model_from_path}" ]; then
    opts+=" --load_model_from_path ${load_model_from_path} "
fi
if [ -n "${resume_from_checkpoint}" ]; then
    opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
fi
if [ -n "${eval_multi_bleu}" ]; then
    opts+=" --eval_multi_bleu ${eval_multi_bleu} "
fi
if [ -n "${no_glm}" ]; then
    opts+=" --no_glm ${no_glm} "
fi
opts+=" --debug ${debug} "
opts+=" --use_asr_prompt ${use_asr_prompt}"
opts+=" --min_promptless_prob ${min_promptless_prob} "
opts+=" --max_promptless_prob ${max_promptless_prob} "
opts+=" --batch_mask_prob ${batch_mask_prob} "
opts+=" --token_mask_prob ${token_mask_prob} "
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
    ["fr"]="test"
    ["all"]="iwslt22_test bbn_cts_bolt_test uhura_test fisher_test callhome_test")

test_set=${testset_dict[${src_lang}]} # This option is to run eval
# test_set="europarl_test"

framework=huggingface # huggingface, openai
preprocessing_num_proc=32
on_the_fly_feat=false

hf_datadir=/exp/cxiao/scale23/hf_data_covost2
datadir=data/${src_lang}
dumpdir=dump_covost2/${src_lang}

if ! "${skip_data_prep}"; then
    # local/prep_covost2.py \
    #     --data-dir ${dumpdir}/raw \
    #     --save-dir ${hf_datadir}/${src_lang}

    # Run Whisper inference on the validation data to create the ASR-prompted validation2 data
    logdir="${dumpdir}/log/validation"
    mkdir -p "${logdir}"
    output_dir="${dumpdir}/decode/validation"
    mkdir -p "${output_dir}"

    # key_file=${hf_datadir}/${src_lang}/validation.wav.scp

    # # 1. Split the key file
    # _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

    # split_scps=""
    # for n in $(seq "${_nj}"); do
    #     split_scps+=" ${logdir}/decode.${n}.scp"
    # done
    # # shellcheck disable=SC2086
    # utils/split_scp.pl "${key_file}" ${split_scps}

    # opts=" --dset ${hf_datadir}/${src_lang}/validation "
    # inference_tool="pyscripts/utils/hf_whisper_inference.py"

    # . ./path_hf.sh
    # . ./cmd.sh

    # if "${debug}"; then
    #     ${inference_tool} \
    #         --keyfile ${logdir}/decode.1.scp \
    #         --src-lang ${src_lang} \
    #         --tgt-lang ${src_lang} \
    #         --output_dir ${logdir}/output.1 \
    #         --batch-size ${inference_batch_size} \
    #         --model_name large-v2 \
    #         --num-beams 1 ${opts}
    # else
    #     # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #     #       but it's used only for deciding the sample ids.
    #     # shellcheck disable=SC2046,SC2086
    #     ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r7n01' --mem 16G --gpu 1 JOB=1:"${_nj}" "${logdir}"/decode.JOB.log \
    #         ${inference_tool} \
    #         --keyfile ${logdir}/decode.JOB.scp \
    #         --src-lang ${src_lang} \
    #         --tgt-lang ${src_lang} \
    #         --output_dir ${logdir}/output.JOB \
    #         --batch-size ${inference_batch_size} \
    #         --model_name large-v2 \
    #         --num-beams 1 ${opts}
    # fi

    # # 3. Concatenates the output files from each jobs
    # mkdir -p "${output_dir}"
    # for i in $(seq "${_nj}"); do
    #     cat "${logdir}/output.${i}/text"
    # done | LC_ALL=C sort -k1 >"${output_dir}/text"

    # Create the validate2 data
    pyscripts/utils/create_synth_data.py \
        --src-dset ${hf_datadir}/${src_lang}/validation \
        --tgt-dset ${hf_datadir}/${src_lang}/validation2 \
        --asr-hyp ${output_dir}/text
fi

if ! "${skip_training}"; then
    ./mml.sh \
        --ngpu 8 \
        --expdir ft_covost2 \
        --nj 80 \
        --st_config ${st_config} \
        --mtl_config ${mtl_config} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --feats_type raw \
        --speed_perturb_factors "0.9 1.0 1.1" \
        --train_set "${train_set}" \
        --mt_train_set "${mt_train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_set}" \
        --stage 1 \
        --stop_stage 1 \
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
        --use_asr_prompt_dev ${use_asr_prompt_dev} \
        --num_beams 1 \
        --score_dir_base scores_ft_covost2 ${opts}
fi
