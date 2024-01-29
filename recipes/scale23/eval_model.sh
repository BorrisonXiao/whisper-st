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

# _modeldir=/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/all/train-cts_sp/mtl/lora
# _modeldir=/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/all/train-cts_sp/st/lora
_modeldir=/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/ara/train-cts_sp/asr/lora
# _modeldir=/home/hltcoe/cxiao/scale23/st/ft_exp/hf_whisper_large-v2_merged/spa/train-all_sp/st/lora
task=st
# src_langs=all
# src_langs="ara cmn rus spa"
# src_langs="ara cmn kor"
# src_langs="rus spa"
src_langs="ara"
logdir=/home/hltcoe/cxiao/scale23/st/logs
outdir=/exp/cxiao/scale23/multi_st_decode
inference_tool=/home/hltcoe/cxiao/scale23/st/pyscripts/utils/hf_whisper_inference.py
inference_batch_size=20
inference_nj=8
merge_decode=true
merge_utt=true
valid_set=dev1
extra_valid_set=dev2
merged_data_base=/exp/cxiao/scale23/merged_data_base
dumpdir=/exp/cxiao/scale23/dump_scale23
feats_type=raw
hf_datadir=/exp/cxiao/scale23/_merged_hf_data
org_hf_datadir=/exp/cxiao/scale23/hf_data
python_hf=/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/hf/bin/python3
evaldir=evaluation
scoredir=/exp/cxiao/scale23/scores_multilingual
sclite_path=sclite
debug=false
eval_cer=false
stage=1
stop_stage=2

. utils/parse_options.sh

. ./path_hf.sh
. ./cmd.sh

# Parse the _modeldir to get the train_set and peft_method
peft_method=${_modeldir##*/}
_path=${_modeldir%/*}
train_obj=${_path##*/}
_path=${_path%/*}
train_set=${_path##*/}
_path=${_path%/*}
train_lang=${_path##*/}
_path=${_path%/*}
model_info=${_path##*/}
model_name=${model_info%_*}
model_name=${model_name#hf_*_}

if [ "${task}" == "asr" ]; then
    merge_decode=true
fi

mtl=false
if [[ "${_modeldir}" == *"/mtl/"* ]]; then
    mtl=true
fi

_mtlprefix=
if "${mtl}"; then
    _mtlprefix=mtl_
fi

decode_suf="_org"
if "${merge_decode}"; then
    decode_suf="_merged"
fi
train_suf="/org"
if "${merge_utt}"; then
    train_suf="/merged"
fi

declare -A testset_dict
testset_dict+=(
    ["ara"]="iwslt22_test"
    ["cmn"]="bbn_cts_bolt_test"
    ["kor"]="uhura_test"
    ["rus"]="uhura_test"
    ["spa"]="fisher_test")

# testset_dict+=(
#     ["ara"]="fleurs_test"
#     ["cmn"]="fleurs_test"
#     ["kor"]="fleurs_test"
#     ["rus"]="fleurs_test"
#     ["spa"]="fleurs_test")

# testset_dict+=(
#     ["ara"]="iwslt22_test fleurs_test"
#     ["cmn"]="bbn_cts_bolt_test fleurs_test"
#     ["kor"]="uhura_test fleurs_test"
#     ["rus"]="uhura_test fleurs_test"
#     ["spa"]="fisher_test callhome_test fleurs_test")

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
            _logdir="${logdir}/inference_${_mtlprefix}${task}/${train_lang}/${src_lang}/${train_set}/${dset}/${peft_method}${train_suf}${decode_suf}"
            mkdir -p "${_logdir}"

            if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
                _suf="/org"
            elif [ "${dset}" = "${train_set}" ]; then
                _suf="/org"
                dset="${train_set}"
            else
                _suf=""
            fi

            if "${merge_decode}"; then
                # If dset is in test_sets, i.e. it contains the "_test" substring, add a suffix to the langdir
                if [[ ${dset} == *"_test" ]]; then
                    _suf="/testsets"
                else
                    _suf=""
                fi

                _srcdir=${merged_data_base}/${src_lang}${_suf}
                _dsetdir=${_logdir}/tmp
                mkdir -p "${_dsetdir}"
                pyscripts/utils/generate_wav_raw.py \
                    -i "${_srcdir}/sr.${src_lang}-${src_lang}.${dset}.stm" \
                    -o "${_dsetdir}"
            else
                _dsetdir=${data_feats}${_suf}/${dset}
                # If the _dsetdir does not exist, run the filter_dev.py script to split the dev into valid_set and extra_valid_set
                if [[ ! -d "${_dsetdir}" && ("${dset}" = "${valid_set}" || "${dset}" = "${extra_valid_set}") ]]; then
                    mkdir -p "${_dsetdir}"
                    _orgdir=${data_feats}${_suf}/dev
                    pyscripts/utils/filter_dev.py \
                        -i "${_orgdir}/wav_raw.scp" \
                        -o "${_dsetdir}/wav_raw.scp" \
                        -r /exp/scale23/data/3-way/${src_lang}/sr.${src_lang}-${src_lang}.${dset}.stm
                fi
            fi

            if [ "${dset}" = "${train_set}" ]; then
                ${python_hf} pyscripts/utils/filter_sp.py \
                    -i "${_dsetdir}/wav_raw.scp" \
                    -o "${_dsetdir}/wav_raw_nosp.scp"

                key_file=${_dsetdir}/wav_raw_nosp.scp
            else
                key_file=${_dsetdir}/wav_raw.scp
            fi

            # 1. Split the key file
            _nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

            split_scps=""
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/decode.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            log "Inference started... log: '${_logdir}/decode.*.log'"
            _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/${_mtlprefix}${task}/${peft_method}${train_suf}${decode_suf}"

            opts=
            if "${merge_decode}"; then
                _hf_dset="${hf_datadir}/${src_lang}.${dset}"
            else
                _hf_dset="${org_hf_datadir}/${src_lang}.${dset}"
            fi
            opts+=" --dset ${_hf_dset} "

            if [ "${peft_method}" != none ]; then
                opts+=" --peft-model ${_modeldir} "
            fi

            if [ "${task}" == "st" ]; then
                opts+=" --task translate "
            fi

            if "${debug}"; then
                ${inference_tool} \
                    --keyfile ${_logdir}/decode.1.scp \
                    --src-lang ${src_lang} \
                    --tgt-lang ${src_lang} \
                    --output_dir ${_logdir}/output.1 \
                    --pretrained-model ${_modeldir} \
                    --batch-size ${inference_batch_size} \
                    --model_name ${model_name} ${opts}
            else
                # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
                #       but it's used only for deciding the sample ids.
                # shellcheck disable=SC2046,SC2086
                ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r7n07' --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                    ${inference_tool} \
                    --keyfile ${_logdir}/decode.JOB.scp \
                    --src-lang ${src_lang} \
                    --tgt-lang ${src_lang} \
                    --output_dir ${_logdir}/output.JOB \
                    --pretrained-model ${_modeldir} \
                    --batch-size ${inference_batch_size} \
                    --model_name ${model_name} ${opts}
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
    log "Stage 2: Run evaluation on the ${task} decoded data."

    for src_lang in ${src_langs}; do
        # If the language is kor, set eval_cer to true
        if [ "${src_lang}" = "kor" ]; then
            eval_cer=true
        else
            eval_cer=false
        fi

        test_sets=${testset_dict[${src_lang}]}
        # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
        for dset in ${test_sets}; do
            # for dset in ${valid_set} ${extra_valid_set}; do
            log "Running ${task} evaluation on ${dset}..."

            _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/${_mtlprefix}${task}/${peft_method}${train_suf}${decode_suf}"
            _dset=$(echo "${dset}" | sed 's/_test$//')

            opts=
            if [ "${src_lang}" == "ara" ]; then
                opts+=" --arabic true "
            fi

            if [ ${task} == "st" ]; then
                if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
                    eval_script=run-devset-eval.sh
                elif [ "${dset}" = "fleurs_test" ]; then
                    eval_script=run-ood-eval.sh
                else
                    eval_script=run-testset-eval.sh
                fi

                opts+=" --hyp_mt ${_dir}/text "
                opts+=" --model_tag ${model_name} "
            elif [ ${task} == "asr" ]; then
                eval_script=run-asr-eval.sh
                opts+=" --cer ${eval_cer} "
                opts+=" --hyp_asr ${_dir}/text "
                opts+=" --sclite ${sclite_path} "
            fi

            if "${merge_decode}"; then
                opts+=" --merge_utt true "
                opts+=" --data_base_dir ${merged_data_base} "
            fi

            cd ${evaldir}
            ${eval_script} \
                --src_lang ${src_lang} \
                --dset "${_dset}" \
                --score_dir ${scoredir}/${_mtlprefix}${task}/hf_whisper_${model_name}/${src_lang}/${peft_method}/${train_set}${train_suf}${decode_suf}/${dset} \
                --framework "huggingface" ${opts}
            cd -
        done
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
