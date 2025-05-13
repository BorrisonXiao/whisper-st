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

model_name=large
src_langs="cmn"
tgt_lang="eng"
logdir=/home/hltcoe/cxiao/scale23/st/logs
outdir=/exp/cxiao/scale23/seamless_decode
inference_batch_size=16
inference_nj=8
dumpdir=/exp/cxiao/scale23/dump_gaussian
feats_type=raw
org_hf_datadir=/exp/cxiao/scale23/hf_data
evaldir=evaluation
scoredir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_seamless
eval_multi_bleu=false
no_glm=true
debug=false
num_beams=1
task=ST
stage=1
stop_stage=2

. utils/parse_options.sh

# It turns out that in queue.pl, ./path.sh is by default sourced, meaning that
# the only way to activate the "correct" path is to use a symbolic link.
# e.g. directly activating ./path_seamless.sh will simply be overwritten by
# ./path.sh after queue.pl is called, which also explains why interactive
# jobs work fine (since they are not using queue.pl, instead just qrsh).
ln -sfv ./path_seamless.sh ./path.sh
. ./path.sh
. ./cmd.sh

export TIKTOKEN_CACHE_DIR=/exp/cxiao/scale23/tiktoken_cache

# The inference_batch_size is re-calculated by dividing by the num_beams
inference_batch_size=$((inference_batch_size / num_beams))
log "inference_batch_size: ${inference_batch_size}"

declare -A testset_dict
testset_dict+=(
    ["ara"]="iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test"
    ["kor"]="uhura_test"
    ["rus"]="uhura_test"
    ["spa"]="fisher_test")

if [ ${src_langs} == "all" ]; then
    src_langs="ara cmn kor rus spa"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for src_lang in ${src_langs}; do
        if [ "${feats_type}" = raw ]; then
            data_feats=${dumpdir}/${src_lang}/raw/
        elif [ "${feats_type}" = fbank_pitch ]; then
            data_feats=${dumpdir}/${src_lang}/fbank_pitch
        elif [ "${feats_type}" = fbank ]; then
            data_feats=${dumpdir}/${src_lang}/fbank
        elif [ "${feats_type}" == extracted ]; then
            data_feats=${dumpdir}/${src_lang}/extracted
        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi

        test_sets=${testset_dict[${src_lang}]}
        # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
        # for dset in ${valid_set} ${extra_valid_set}; do
        for dset in ${test_sets}; do
            log "Running inference for ${src_lang} ${dset}"
            _logdir="${logdir}/inference_seamless/${task}/${src_lang}/${dset}"
            mkdir -p "${_logdir}"

            _dsetdir=${data_feats}${dset}
            # If the _dsetdir does not exist, run the filter_dev.py script to split the dev into valid_set and extra_valid_set
            if [[ ! -d "${_dsetdir}" ]]; then
                mkdir -p "${_dsetdir}"
                _orgdir=${data_feats}/dev
                pyscripts/utils/filter_dev.py \
                    -i "${_orgdir}/wav_raw.scp" \
                    -o "${_dsetdir}/wav_raw.scp" \
                    -r /exp/scale23/data/3-way/${src_lang}/sr.${src_lang}-${src_lang}.${dset}.stm
            fi

            key_file=${_dsetdir}/wav_raw.scp

            # 1. Split the key file
            _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

            split_scps=""
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/decode.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            log "Inference started... log: '${_logdir}/decode.*.log'"
            _dir="${outdir}/${src_lang}/${dset}/${task}"

            opts=
            _hf_dset="${org_hf_datadir}/${src_lang}.${dset}"
            opts+=" --dset ${_hf_dset} "

            inference_tool="pyscripts/utils/seamless_inference.py"

            if "${debug}"; then
                ${inference_tool} \
                    --keyfile ${_logdir}/decode.1.scp \
                    --src-lang ${src_lang} \
                    --task ${task} \
                    --tgt-lang ${tgt_lang} \
                    --output_dir ${_logdir}/output.1 \
                    --batch-size ${inference_batch_size} \
                    --num-beams ${num_beams} \
                    --model-name ${model_name} ${opts}
            else
                # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
                #       but it's used only for deciding the sample ids.
                # shellcheck disable=SC2046,SC2086
                ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r8n06\&!r9n02' --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                    ${inference_tool} \
                    --keyfile ${_logdir}/decode.JOB.scp \
                    --task ${task} \
                    --src-lang ${src_lang} \
                    --tgt-lang ${tgt_lang} \
                    --output_dir ${_logdir}/output.JOB \
                    --batch-size ${inference_batch_size} \
                    --num-beams ${num_beams} \
                    --model-name ${model_name} ${opts}
            fi

            # 3. Concatenates the output files from each jobs
            mkdir -p "${_dir}"
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/text"
            done | LC_ALL=C sort -k1 >"${_dir}/text"
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Run evaluation on the decoded data."

    for src_lang in ${src_langs}; do
        test_sets=${testset_dict[${src_lang}]}
        # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
        for dset in ${test_sets}; do
            # for dset in ${valid_set} ${extra_valid_set}; do

            _dir="${outdir}/${src_lang}/${dset}/${task}"
            _dset=$(echo "${dset}" | sed 's/_test$//')

            opts=
            if [ "${src_lang}" == "ara" ]; then
                opts+=" --arabic true "
            fi

            eval_script=run-testset-eval.sh

            if [ "${task}" == "ASR" ]; then
                eval_script=run-asr-eval.sh
                opts+=" --hyp_asr ${_dir}/text "
                opts+=" --sclite sclite "
            else
                opts+=" --hyp_mt ${_dir}/text "
                opts+=" --model_tag ${model_name} "
                if "${eval_multi_bleu}"; then
                    opts+=" --eval-multi-bleu true "
                fi

                if "${no_glm}"; then
                    opts+=" --no-glm true "
                fi
            fi

            _scoredir=${scoredir}/${model_name}/${task}/${src_lang}/${dset}

            cd ${evaldir}
            ${eval_script} \
                --src_lang ${src_lang} \
                --dset "${_dset}" \
                --score_dir "${_scoredir}" \
                --framework "huggingface" ${opts}
            cd -
        done
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
