#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' task, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Run inference for a model on a specific test set

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

model=large-v2
task=st
data_base_dir=/home/cxiao7/research/legicost/export
expdir=exp/${model}
logdir=${expdir}/logs
outdir=${expdir}/decode
inference_tool=pyscripts/utils/hf_whisper_inference.py
inference_batch_size=8
inference_nj=4
valid_set=dev-asr-0
extra_valid_set=dev-mt-0
dumpdir=dump
evaldir=evaluation
scoredir=${expdir}/scores
sclite_path=/home/cxiao7/research/espnet-st/tools/sctk-20159b5/bin/sclite
debug=false
eval_cer=false
test_sets="test"
src_lang="cmn"
stage=1
stop_stage=2

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # for dset in ${test_sets}; do
    for dset in ${valid_set}; do
    # for dset in ${extra_valid_set}; do
        log "Running inference for ${dset}"
        _logdir="${logdir}/inference_${task}/${dset}/"
        mkdir -p "${_logdir}"

        key_file=${dumpdir}/raw/${dset}/uttids

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        log "Inference started... log: '${_logdir}/decode.*.log'"
        _dir="${outdir}/${dset}/${task}"

        opts=
        opts+=" --dset ${dumpdir}/raw/${dset} "

        if [ "${task}" == "st" ]; then
            opts+=" --task translate "
        fi

        if "${debug}"; then
            ${inference_tool} \
                --keyfile ${_logdir}/decode.1.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${src_lang} \
                --output_dir ${_logdir}/output.1 \
                --batch-size ${inference_batch_size} \
                --model_name ${model} ${opts}
        else
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2046,SC2086
            ${cuda_cmd} --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                ${inference_tool} \
                --keyfile ${_logdir}/decode.JOB.scp \
                --src-lang ${src_lang} \
                --tgt-lang ${src_lang} \
                --output_dir ${_logdir}/output.JOB \
                --batch-size ${inference_batch_size} \
                --model_name ${model} ${opts}
        fi

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Run evaluation on the ${task} decoded data."

    # for dset in ${test_sets}; do
    for dset in ${valid_set}; do
    # for dset in ${extra_valid_set}; do
        log "Running ${task} evaluation on ${dset}..."

        _dir="${outdir}/${dset}/${task}"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts="--data_base_dir ${data_base_dir} "

        if [ ${task} == "st" ]; then
            eval_script=run-testset-eval.sh
            opts+=" --hyp_mt ${PWD}/${_dir}/text "
        elif [ ${task} == "asr" ]; then
            eval_script=run-asr-eval.sh
            opts+=" --cer ${eval_cer} "
            opts+=" --hyp_asr ${PWD}/${_dir}/text "
            opts+=" --sclite ${sclite_path} "
        fi

        cd ${evaldir}
        ${PWD}/${eval_script} \
            --src_lang ${src_lang} \
            --dset "${_dset}" \
            --score_dir ${PWD}/${scoredir}/${task}/hf_whisper_${model}/${src_lang}/${dset} \
            --framework "huggingface" ${opts}
        cd -
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
