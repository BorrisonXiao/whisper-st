#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
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
SECONDS=0

# Evaluation related
sclite_path=sclite

# General configuration
ds_config=
mt_config=
datadir=
stage=1                  # Processes starts from the specified stage.
stop_stage=1             # Processes is stopped at the specified stage.
skip_train=false         # Skip training stages.
skip_eval=false          # Skip decoding and evaluation stages.
ngpu=1                   # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1              # The number of nodes.
nj=32                    # The number of parallel jobs.
inference_nj=32          # The number of parallel jobs in decoding.
dumpdir=dump             # Directory to dump features.
expdir=nllb_exp          # Directory to save experiments.
python=python3           # Specify python to execute espnet commands.
model_name=1.3B          # Model name, e.g. "base", "large", etc.
hf_datadir=              # Directory to the hugging face dataset.
src_lang=es              # source language abbrev. id (e.g., es)
tgt_lang=en              # target language abbrev. id (e.g., en)
debug=false              # If true, only one batch is processed in inference.
peft_method=none         # none, lora, qlora
preprocessing_num_proc=8 # Number of processes for preprocessing
master_port=29501        # Master port for distributed training (to avoid conflict on the same node)
resume_from_checkpoint=  # The path to resume from a checkpoint
load_model_from_path=    # The path to load the model from
merged_data_base=        # The base directory for the merged data
merged_data_dir=         # The directory for the merged data

# [Task dependent] Set the datadir name created by local/data.sh
train_set=         # Name of training set.
valid_set=         # Name of validation set used for monitoring/tuning network training.
extra_valid_set="" # Name of extra validation set used for evaluation.
test_sets=         # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

inference_batch_size=8
num_beams=1
inference_checkpoint= # The checkpoint to use for inference.

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh
. ./path_hf.sh
. ./cmd.sh

data_feats=${dumpdir}/raw

# ========================== Main stages start from here. ==========================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Run (distributed) MT inference on the dev/test data."
    # for dset in ${valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    for dset in ${test_sets}; do
        _dsetdir=${data_feats}/${dset}
        _dir="${expdir}/${src_lang}/${dset}"
        _logdir="${expdir}/logdir/inference/${src_lang}/${dset}"
        mkdir -p "${_logdir}"

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${_dsetdir}/text -l)")

        key_file=${_dsetdir}/text
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.text"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Inference started... log: '${_logdir}/decode.*.log'"

        opts=
        inference_tool="pyscripts/utils/nllb_inference.py"

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.text \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.1 \
                --batch-size ${inference_batch_size} \
                --num-beams ${num_beams} \
                --model_name ${model_name} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${cuda_cmd} --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.text \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.JOB \
                --batch-size ${inference_batch_size} \
                --num-beams ${num_beams} \
                --model_name ${model_name} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Run evaluation on the MT decoded data."

    # Note that we assume the evaluation code is available in the path
    # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    for dset in ${test_sets}; do
        # for dset in ${extra_valid_set} ${test_sets}; do
        # for dset in ${extra_valid_set}; do
        log "Running MT evaluation on ${dset}"
        if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
            eval_script=run-devset-eval.sh
        elif [ "${dset}" = "fleurs_test" ]; then
            eval_script=run-ood-eval.sh
        else
            eval_script=run-testset-eval.sh
        fi

        _dir="${expdir}/${src_lang}/${dset}"
        _st_hyp="${PWD}/${_dir}/text"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts=
        score_dir=scores/nllb/${model_name}/${src_lang}/vanilla/${dset}

        cd evaluation
        ${eval_script} \
            --src_lang ${src_lang} \
            --hyp_mt "${_st_hyp}" \
            --model_tag ${model_name} \
            --dset "${_dset}" \
            --score_dir "${score_dir}" \
            --framework "huggingface" ${opts}
        cd -
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Run MT finetuning on the training data"
    _dir="${expdir}/${src_lang}/${train_set}/mt/${peft_method}"
    _logdir="${_dir}/logdir"
    mkdir -p "${_logdir}"

    text_dir=${datadir}

    opts=" --input-dir ${text_dir} "
    opts+=" --hf-datadir ${hf_datadir} "
    if "${debug}"; then
        opts+=" --preprocessing_num_proc 1 "
    else
        opts+=" --preprocessing_num_proc ${preprocessing_num_proc} "
    fi
    opts+=" --dev-set ${extra_valid_set} "

    if [ -n "${mt_config}" ]; then
        opts+=" --config ${mt_config} "
    fi

    if [ "${peft_method}" != none ]; then
        opts+=" --peft_method ${peft_method} "
    fi

    train_tool="pyscripts/utils/nllb_ft.py"

    if [ -n "${resume_from_checkpoint}" ]; then
        opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
    fi
    if [ -n "${load_model_from_path}" ]; then
        opts+=" --load_model_from_path ${load_model_from_path} "
    fi
    if [ -n "${ds_config}" ]; then
        opts+=" --deepspeed ${ds_config} "
    fi

    # Submit the training jobs
    JOBID=$(date +'%Y%m%d%H%M%S')
    log "Training started... log: '${PWD}/${_logdir}/finetune_${JOBID}.log'"

    if "${debug}"; then
        ${python} ${train_tool} \
            --train-set ${train_set} \
            --src-lang ${src_lang} \
            --tgt-lang ${tgt_lang} \
            --output_dir ${_dir} \
            --model_name ${model_name} ${opts}
    else
        # For some reason the node r9n01 is much faster than the other nodes
        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        # shellcheck disable=SC2046,SC2086
        # ${cuda_cmd} --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
        # ${cuda_cmd} --hostname 'r9n03' --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
        ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06' --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
            ${python} -m torch.distributed.launch --nproc_per_node ${ngpu} --master_port ${master_port} \
            ${train_tool} \
            --train-set ${train_set} \
            --src-lang ${src_lang} \
            --tgt-lang ${tgt_lang} \
            --output_dir ${_dir} \
            --model_name ${model_name} ${opts}
    fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Run (distributed) MT inference on the dev/test data."
    # for dset in ${valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    for dset in ${test_sets}; do
        # If dset is in test_sets, i.e. it contains the "_test" substring, add a suffix to the langdir
        if [[ ${dset} == *"_test" ]]; then
            _suf="/testsets"
        else
            _suf=""
        fi

        _logdir="${expdir}/logdir/inference/mt/${src_lang}/${train_set}/${dset}/${peft_method}"
        mkdir -p "${_logdir}"

        _srcdir=${merged_data_base}/${src_lang}${_suf}
        _dsetdir=${_logdir}/tmp
        mkdir -p "${_dsetdir}"
        pyscripts/utils/generate_wav_raw.py \
            -i "${_srcdir}/st.${src_lang}-${tgt_lang}.${dset}.stm" \
            -o "${_dsetdir}"

        _dsetdir=${data_feats}/${dset}
        _dir="${expdir}/${src_lang}/decode/${train_set}/${dset}/${peft_method}"
        _modeldir="${expdir}/${src_lang}/${train_set}/mt/${peft_method}"

        if [ -n "${inference_checkpoint}" ]; then
            _modeldir="${_modeldir}/${inference_checkpoint}"
            _dir="${expdir}/${src_lang}/decode/${train_set}/${dset}/${peft_method}_${inference_checkpoint}"
        fi

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${_dsetdir}/text -l)")

        key_file=${_dsetdir}/text
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.text"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Inference started... log: '${_logdir}/decode.*.log'"

        opts=

        inference_tool="pyscripts/utils/nllb_inference.py"

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.text \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.1 \
                --batch-size ${inference_batch_size} \
                --pretrained-model ${_modeldir} \
                --num-beams ${num_beams} \
                --model_name ${model_name} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${cuda_cmd} --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.text \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.JOB \
                --batch-size ${inference_batch_size} \
                --pretrained-model ${_modeldir} \
                --num-beams ${num_beams} \
                --model_name ${model_name} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Run evaluation on the MT decoded data."

    # Note that we assume the evaluation code is available in the path
    # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    for dset in ${test_sets}; do
        # for dset in ${extra_valid_set} ${test_sets}; do
        # for dset in ${extra_valid_set}; do
        log "Running MT evaluation on ${dset}"
        if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
            eval_script=run-devset-eval.sh
        elif [ "${dset}" = "fleurs_test" ]; then
            eval_script=run-ood-eval.sh
        else
            eval_script=run-testset-eval.sh
        fi

        _dir="${expdir}/${src_lang}/decode/${train_set}/${dset}/${peft_method}"
        if [ -n "${inference_checkpoint}" ]; then
            _dir="${expdir}/${src_lang}/decode/${train_set}/${dset}/${peft_method}_${inference_checkpoint}"
        fi
        _st_hyp="${PWD}/${_dir}/text"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts=
        score_dir=scores/nllb/${model_name}/${src_lang}/${train_set}/${dset}

        cd evaluation
        ${eval_script} \
            --src_lang ${src_lang} \
            --hyp_mt "${_st_hyp}" \
            --model_tag ${model_name} \
            --dset "${_dset}" \
            --score_dir "${score_dir}" \
            --framework "huggingface" ${opts}
        cd -
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Run MT finetuning on the merged training data"
    _dir="${expdir}/${src_lang}/${train_set}_merged/mt/${peft_method}"
    _logdir="${_dir}/logdir"
    _tmpdir="${_logdir}/tmp"
    for dset in ${train_set} ${extra_valid_set}; do
        _dsetdir=${_tmpdir}/${dset}
        mkdir -p "${_dsetdir}"
        pyscripts/utils/stm2text.py \
            -i "${merged_data_base}/${src_lang}/st.${src_lang}-${tgt_lang}.${dset}.stm" \
            -o "${_dsetdir}/text.tc.${tgt_lang}"
        pyscripts/utils/stm2text.py \
            -i "${merged_data_base}/${src_lang}/sr.${src_lang}-${src_lang}.${dset}.stm" \
            -o "${_dsetdir}/text.tc.${src_lang}"
    done

    text_dir=${_tmpdir}

    opts=" --input-dir ${text_dir} "
    opts+=" --hf-datadir ${merged_data_dir} "
    if "${debug}"; then
        opts+=" --preprocessing_num_proc 1 "
    else
        opts+=" --preprocessing_num_proc ${preprocessing_num_proc} "
    fi
    opts+=" --dev-set ${extra_valid_set} "

    if [ -n "${mt_config}" ]; then
        opts+=" --config ${mt_config} "
    fi

    if [ "${peft_method}" != none ]; then
        opts+=" --peft_method ${peft_method} "
    fi

    train_tool="pyscripts/utils/nllb_ft.py"

    if [ -n "${resume_from_checkpoint}" ]; then
        opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
    fi
    if [ -n "${load_model_from_path}" ]; then
        opts+=" --load_model_from_path ${load_model_from_path} "
    fi
    if [ -n "${ds_config}" ]; then
        opts+=" --deepspeed ${ds_config} "
    fi

    # Submit the training jobs
    JOBID=$(date +'%Y%m%d%H%M%S')
    log "Training started... log: '${PWD}/${_logdir}/finetune_${JOBID}.log'"

    if "${debug}"; then
        ${python} ${train_tool} \
            --train-set ${train_set} \
            --src-lang ${src_lang} \
            --tgt-lang ${tgt_lang} \
            --output_dir ${_dir} \
            --model_name ${model_name} ${opts}
    else
        # For some reason the node r9n01 is much faster than the other nodes
        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        # shellcheck disable=SC2046,SC2086
        # ${cuda_cmd} --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
        # ${cuda_cmd} --hostname 'r9n03' --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
        ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06' --mem 16G --gpu ${ngpu} "${_logdir}"/finetune_${JOBID}.log \
            ${python} -m torch.distributed.launch --nproc_per_node ${ngpu} --master_port ${master_port} \
            ${train_tool} \
            --train-set ${train_set} \
            --src-lang ${src_lang} \
            --tgt-lang ${tgt_lang} \
            --output_dir ${_dir} \
            --model_name ${model_name} ${opts}
    fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Run (distributed) MT inference on the dev/test data."
    # for dset in ${valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    for dset in ${test_sets}; do
        # If dset is in test_sets, i.e. it contains the "_test" substring, add a suffix to the langdir
        if [[ ${dset} == *"_test" ]]; then
            _suf="/testsets"
        else
            _suf=""
        fi

        _logdir="${expdir}/logdir/inference/mt/${src_lang}/${train_set}_merged/${dset}/${peft_method}"
        mkdir -p "${_logdir}"

        _srcdir=${merged_data_base}/${src_lang}${_suf}
        _dsetdir=${_logdir}/tmp
        mkdir -p "${_dsetdir}"
        pyscripts/utils/generate_wav_raw.py \
            -i "${_srcdir}/st.${src_lang}-${tgt_lang}.${dset}.stm" \
            -o "${_dsetdir}"

        _dsetdir=${data_feats}/${dset}
        _dir="${expdir}/${src_lang}/decode/${train_set}_merged/${dset}/${peft_method}"
        _modeldir="${expdir}/${src_lang}/${train_set}_merged/mt/${peft_method}"

        if [ -n "${inference_checkpoint}" ]; then
            _modeldir="${_modeldir}/${inference_checkpoint}"
            _dir="${expdir}/${src_lang}/decode/${train_set}_merged/${dset}/${peft_method}_${inference_checkpoint}"
        fi

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${_dsetdir}/text -l)")

        key_file=${_dsetdir}/text
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.text"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Inference started... log: '${_logdir}/decode.*.log'"

        opts=

        inference_tool="pyscripts/utils/nllb_inference.py"

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.text \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.1 \
                --batch-size ${inference_batch_size} \
                --pretrained-model ${_modeldir} \
                --num-beams ${num_beams} \
                --model_name ${model_name} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${cuda_cmd} --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.text \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_logdir}/output.JOB \
                --batch-size ${inference_batch_size} \
                --pretrained-model ${_modeldir} \
                --num-beams ${num_beams} \
                --model_name ${model_name} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Run evaluation on the MT decoded data."

    # Note that we assume the evaluation code is available in the path
    # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    for dset in ${test_sets}; do
        # for dset in ${extra_valid_set} ${test_sets}; do
        # for dset in ${extra_valid_set}; do
        log "Running MT evaluation on ${dset}"
        if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
            eval_script=run-devset-eval.sh
        elif [ "${dset}" = "fleurs_test" ]; then
            eval_script=run-ood-eval.sh
        else
            eval_script=run-testset-eval.sh
        fi

        _dir="${expdir}/${src_lang}/decode/${train_set}_merged/${dset}/${peft_method}"
        if [ -n "${inference_checkpoint}" ]; then
            _dir="${expdir}/${src_lang}/decode/${train_set}_merged/${dset}/${peft_method}_${inference_checkpoint}"
        fi
        _st_hyp="${PWD}/${_dir}/text"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts=
        score_dir=scores/nllb/${model_name}/${src_lang}/${train_set}_merged/${dset}

        cd evaluation
        ${eval_script} \
            --src_lang ${src_lang} \
            --hyp_mt "${_st_hyp}" \
            --model_tag ${model_name} \
            --dset "${_dset}" \
            --score_dir "${score_dir}" \
            --framework "huggingface" ${opts}
        cd -
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
