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

_modeldir=/home/hltcoe/cxiao/st/ft_dgx/hf_whisper_large-v2_merged/spa/train-cts_sp/mml/lora_0.8_0.2/checkpoint-7600
src_langs="spa"
logdir=/home/hltcoe/cxiao/scale23/st/logs
outdir=/exp/cxiao/scale23/multi_st_decode/mml_ood
inference_batch_size=16
inference_nj=8
merge_utt=true
dumpdir=/exp/cxiao/scale23/dump_gaussian
feats_type=raw
org_hf_datadir=/exp/cxiao/scale23/hf_data
python_hf=python3
evaldir=evaluation
scoredir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_tmp/mml_ood/st/hf_whisper_large-v2/spa/lora_0.8_0.2_checkpoint-7600/train-cts_sp/merged_org/fisher_test
use_asr_prompt=true
eval_multi_bleu=false
no_glm=true
debug=true
num_beams=1
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

# The inference_batch_size is re-calculated by dividing by the num_beams
inference_batch_size=$((inference_batch_size / num_beams))
log "inference_batch_size: ${inference_batch_size}"

decode_suf="_org"
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
            _logdir="${logdir}/inference_mml/st/${train_lang}/${src_lang}/${train_set}/${dset}/${peft_method}${train_suf}${decode_suf}"
            mkdir -p "${_logdir}"

            if [ "${dset}" = "${train_set}" ]; then
                _suf="/org"
                dset="${train_set}"
            else
                _suf=""
            fi

            _dsetdir=${data_feats}${_suf}/${dset}
            # If the _dsetdir does not exist, run the filter_dev.py script to split the dev into valid_set and extra_valid_set
            if [[ ! -d "${_dsetdir}" ]]; then
                mkdir -p "${_dsetdir}"
                _orgdir=${data_feats}${_suf}/dev
                pyscripts/utils/filter_dev.py \
                    -i "${_orgdir}/wav_raw.scp" \
                    -o "${_dsetdir}/wav_raw.scp" \
                    -r /exp/scale23/data/3-way/${src_lang}/sr.${src_lang}-${src_lang}.${dset}.stm
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
            _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/st/${peft_method}${train_suf}${decode_suf}"
            if "${use_asr_prompt}"; then
                _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/st/${peft_method}${train_suf}${decode_suf}_asr_prompt"
            fi

            opts=
            _hf_dset="${org_hf_datadir}/${src_lang}.${dset}"
            opts+=" --dset ${_hf_dset} "

            if [ "${peft_method}" != none ]; then
                opts+=" --peft-model ${_modeldir} "
            fi

            inference_tool="pyscripts/utils/hf_whisper_inference_pmtl.py"
            if "${use_asr_prompt}"; then
                opts+=" --use-asr-hyp "
                opts+=" --asr-hyp ${_dir}/asr "
            else
                opts+=" --disable-asr "
            fi

            if "${debug}"; then
                ${inference_tool} \
                    --keyfile ${_logdir}/decode.1.scp \
                    --src-lang ${src_lang} \
                    --tgt-lang ${src_lang} \
                    --output_dir ${_logdir}/output.1 \
                    --pretrained-model ${_modeldir} \
                    --batch-size ${inference_batch_size} \
                    --num-beams ${num_beams} \
                    --model_name ${model_name} ${opts}
            else
                # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
                #       but it's used only for deciding the sample ids.
                # shellcheck disable=SC2046,SC2086
                ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r8n06\&!r9n02' --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
                    ${inference_tool} \
                    --keyfile ${_logdir}/decode.JOB.scp \
                    --src-lang ${src_lang} \
                    --tgt-lang ${src_lang} \
                    --output_dir ${_logdir}/output.JOB \
                    --pretrained-model ${_modeldir} \
                    --batch-size ${inference_batch_size} \
                    --num-beams ${num_beams} \
                    --model_name ${model_name} ${opts}
            fi

            # 3. Concatenates the output files from each jobs
            mkdir -p "${_dir}"
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/st"
            done | LC_ALL=C sort -k1 >"${_dir}/st"
            if "${use_asr_prompt}"; then
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/asr"
                done | LC_ALL=C sort -k1 >"${_dir}/asr"
            fi
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Run evaluation on the decoded data."

    for src_lang in ${src_langs}; do
        test_sets=${testset_dict[${src_lang}]}
        # Run ASR evaluation
        if "${use_asr_prompt}"; then
            # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
            # for dset in ${valid_set}; do
            for dset in ${test_sets}; do
                # for dset in ${extra_valid_set}; do
                # for dset in ${extra_valid_set} ${test_sets}; do
                log "Running ASR evaluation on ${dset}"
                eval_script=run-asr-eval.sh

                _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/st/${peft_method}${train_suf}${decode_suf}_asr_prompt"
                _asr_hyp="${_dir}/asr"
                _dset=$(echo "${dset}" | sed 's/_test$//')

                opts=
                if [ "${src_lang}" == "ara" ]; then
                    opts+=" --arabic true "
                fi

                _scoredir=${scoredir}_asr_prompt_asr

                cd evaluation
                ${eval_script} \
                    --src_lang ${src_lang} \
                    --hyp_asr "${_asr_hyp}" \
                    --sclite sclite \
                    --dset "${_dset}" \
                    --score_dir "${_scoredir}" \
                    --framework "huggingface" ${opts}
                cd -
            done
        fi
    done

    for src_lang in ${src_langs}; do
        test_sets=${testset_dict[${src_lang}]}
        # for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
        for dset in ${test_sets}; do
            # for dset in ${valid_set} ${extra_valid_set}; do

            _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/st/${peft_method}${train_suf}${decode_suf}"
            if "${use_asr_prompt}"; then
                _dir="${outdir}/${train_lang}/${src_lang}/${train_set}/${dset}/st/${peft_method}${train_suf}${decode_suf}_asr_prompt"
            fi
            _dset=$(echo "${dset}" | sed 's/_test$//')

            opts=
            if [ "${src_lang}" == "ara" ]; then
                opts+=" --arabic true "
            fi

            eval_script=run-testset-eval.sh

            opts+=" --hyp_mt ${_dir}/st "
            opts+=" --model_tag ${model_name} "

            _scoredir=${scoredir}
            if "${use_asr_prompt}"; then
                _scoredir=${scoredir}_asr_prompt
            fi
            if "${eval_multi_bleu}"; then
                opts+=" --eval-multi-bleu true "
            fi

            if "${no_glm}"; then
                opts+=" --no-glm true "
            fi

            cd ${evaldir}
            ${eval_script} \
                --src_lang ${src_lang} \
                --model_tag ${model_name} \
                --dset "${_dset}" \
                --score_dir "${_scoredir}" \
                --framework "huggingface" ${opts}
            cd -
        done
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
