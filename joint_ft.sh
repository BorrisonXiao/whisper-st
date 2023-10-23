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
stage=1                                   # Processes starts from the specified stage.
stop_stage=10000                          # Processes is stopped at the specified stage.
ngpu=1                                    # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1                               # The number of nodes.
nj=32                                     # The number of parallel jobs.
inference_nj=32                           # The number of parallel jobs in decoding.
gpu_inference=false                       # Whether to perform gpu decoding.
dumpdir=dump                              # Directory to dump features.
expdir=exp                                # Directory to save experiments.
python=python3                            # Specify python to execute espnet commands.
model_name=base                           # Model name, e.g. "base", "large", etc.
framework=huggingface                     # huggingface, openai
hf_datadir=                               # Directory to the hugging face dataset.
org_hf_datadir=/exp/cxiao/scale23/hf_data # Directory to the original hugging face dataset.
preprocessing_num_proc=4                  # Number of parallel jobs in preprocessing
resume_from_checkpoint=                   # Resume from checkpoint path
peft_method=none                          # none, lora, qlora
on_the_fly_feat=false                     # Whether to generate features on the fly
debug=false                               # Whether to use debug mode
dev_name=dev                              # Name of the dev set, e.g. dev, dev1, dev2
precompute_feats=true                     # Whether to precompute features (useful for multi-gpu training)
ds_config=                                # Path to the deepspeed config file
asr_save_eval_preds=                      # Path to store the asr evaluation predictions for analysis
st_save_eval_preds=                       # Path to store the st evaluation predictions for analysis
master_port=29500                         # Port for distributed training
merge_utt=false                           # Whether to merge utterances to the closest 30s for training and inference
merged_data_base=                         # Base directory for merged data
normalize_text=false                      # Whether to normalize text before training and during validation
python_hf=python3                         # Specify python to execute hugging face commands.
fe_only=false                             # Whether to do feature extraction only
asr_config=                               # Config for asr model training.
mtl_config=                               # Config for multi-task model training.
eval_cer=false                            # Whether to evaluate CER
inference_batch_size=32                   # Batch size for inference
merge_decode=true                         # Whether to decode on the merged data
dialect=                                  # The dialect language code will be used instead of the src_lang code if specified

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors= # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw # Feature type (raw or fbank_pitch).

# ST model related
st_tag= # Suffix to the result dir for st model training.
st_exp= # Specify the directory path for ST experiment.
# If this option is specified, st_tag is ignored.
st_config= # Config for st model training.
# Note that it will overwrite args in st config.
src_lang=es # source language abbrev. id (e.g., es)
tgt_lang=en # target language abbrev. id (e.g., en)

# [Task dependent] Set the datadir name created by local/data.sh
train_set=         # Name of training set.
valid_set=         # Name of validation set used for monitoring/tuning network training.
extra_valid_set="" # Name of extra validation set used for evaluation.
test_sets=         # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").

    # ST model related
    --st_tag           # Suffix to the result dir for st model training (default="${st_tag}").
    --st_exp           # Specify the directory path for ST experiment.
                       # If this option is specified, st_tag is ignored (default="${st_exp}").
    --st_config        # Config for st model training (default="${st_config}").
    --src_lang=        # source language abbrev. id (e.g., es). (default="${src_lang}")
    --tgt_lang=        # target language abbrev. id (e.g., en). (default="${tgt_lang}")
    --use_src_lang=    # Incorporate ASR loss (use src texts) or not 
    
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

if [ "${framework}" == "huggingface" ]; then
    . ./path_hf.sh
else
    . ./path.sh
fi
. ./cmd.sh

# Check required arguments
[ -z "${train_set}" ] && {
    log "${help_message}"
    log "Error: --train_set is required"
    exit 2
}
[ -z "${valid_set}" ] && {
    log "${help_message}"
    log "Error: --valid_set is required"
    exit 2
}
[ -z "${test_sets}" ] && {
    log "${help_message}"
    log "Error: --test_sets is required"
    exit 2
}

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# The directory used for training commands
if [ -z "${st_exp}" ]; then
    if "${merge_utt}"; then
        _suf="_merged"
    else
        _suf=""
    fi

    if [ "${framework}" = "huggingface" ]; then
        st_exp="${expdir}/hf_${st_tag}${_suf}"
    else
        st_exp="${expdir}/${st_tag}${_suf}"
    fi
fi

# ========================== Main stages start from here. ==========================

# Re-check if the training set needs the "_sp" suffix, note that this is to
# accommodate the "skip_data_prep" mode, i.e. if the suffix is there already,
# don't add it again.
if [ -n "${speed_perturb_factors}" ] && ! echo "${train_set}" | grep -q "_sp"; then
    train_set="${train_set}_sp"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Run Bayesian-decomposed multitask finetuning on the training data"
    _dir="${st_exp}/${src_lang}/${train_set}/bmtl/${peft_method}"
    _logdir="${_dir}/logdir"
    mkdir -p "${_logdir}"

    opts=" --mode bmtl "
    if [ "${framework}" == "huggingface" ]; then
        opts+=" --hf_datadir ${hf_datadir} "
        if "${debug}"; then
            opts+=" --preprocessing_num_proc 1 "
        else
            opts+=" --preprocessing_num_proc ${preprocessing_num_proc} "
        fi
        opts+=" --dev-name ${extra_valid_set} "

        if [ -n "${st_config}" ]; then
            opts+=" --config ${mtl_config} "
        fi

        if [ "${peft_method}" != none ]; then
            opts+=" --peft_method ${peft_method} "
        fi

        _feat_type=feats
        if "${on_the_fly_feat}"; then
            opts+=" --on-the-fly-feat-extraction "
            _feat_type=raw
        fi

        if "${normalize_text}"; then
            opts+=" --normalize_text "
        fi

        opts+=" --save_feature_dir ${hf_datadir}/features/${_feat_type} "

        train_tool="pyscripts/utils/hf_whisper_ft.py"
    else
        log "Error: not supported --framework ${framework}"
        exit 2
    fi
    if [ -n "${resume_from_checkpoint}" ]; then
        opts+=" --resume_from_checkpoint ${resume_from_checkpoint} "
    fi
    if [ -n "${ds_config}" ]; then
        opts+=" --deepspeed ${ds_config} "
    fi
    # if [ -n "${st_save_eval_preds}" ]; then
    #     opts+=" --save-eval-preds ${st_save_eval_preds} "
    # fi

    if "${precompute_feats}"; then
        # If the src_lang is "all", check for all the languages, including ara, cmn, kor, rus and spa
        if [ "${src_lang}" == "all" ]; then
            log "Currently does not support multilingual training..."
            exit 2
        else
            # If the feature is already extracted in previous runs, skip this step
            if [ ! -d "${hf_datadir}/features/${_feat_type}/${src_lang}.${train_set}.bmtl" ] ||
                [ ! -d "${hf_datadir}/features/${_feat_type}/${src_lang}.${extra_valid_set}.bmtl" ]; then
                log "${hf_datadir}/features/${_feat_type}/${src_lang}.${train_set}.bmtl or ${hf_datadir}/features/${_feat_type}/${src_lang}.${extra_valid_set}.bmtl does not exist..."
                if "${debug}"; then
                    ${python_hf} ${train_tool} \
                        --feat-extraction \
                        --train-set ${train_set} \
                        --src-lang ${src_lang} \
                        --tgt-lang ${tgt_lang} \
                        --output_dir ${_dir} \
                        --model_name ${model_name} ${opts}
                else
                    # Submit the feature extraction jobs
                    JOBID=$(date +'%Y%m%d%H%M%S')
                    log "${hf_datadir}/features/${_feat_type}/${src_lang}.${train_set}.bmtl or ${hf_datadir}/features/${_feat_type}/${src_lang}.${extra_valid_set}.bmtl does not exist..."
                    log "Feature extraction started... log: '${_logdir}/fe_${JOBID}.log'"
                    ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06' --mem 64G --gpu 1 "${_logdir}"/fe_${JOBID}.log \
                        ${python_hf} ${train_tool} \
                        --feat-extraction \
                        --train-set ${train_set} \
                        --src-lang ${src_lang} \
                        --tgt-lang ${tgt_lang} \
                        --output_dir ${_dir} \
                        --model_name ${model_name} ${opts}
                fi
            else
                log "Skip feature extraction as the features are already extracted"
            fi
        fi
    fi

    if "${fe_only}"; then
        log "Skip training as --fe_only is set to true"
    else
        # Submit the training jobs
        JOBID=$(date +'%Y%m%d%H%M%S')
        log "Training started... log: '${_logdir}/finetune_${JOBID}.log'"

        if "${debug}"; then
            ${python_hf} ${train_tool} \
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
                ${python_hf} -m torch.distributed.launch --nproc_per_node ${ngpu} --master_port ${master_port} \
                ${train_tool} \
                --train-set ${train_set} \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --output_dir ${_dir} \
                --model_name ${model_name} ${opts}
        fi
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Run (distributed) MTL inference on the dev/test data."
    decode_suf="_org"
    if "${merge_decode}"; then
        decode_suf="_merged"
    fi
    train_suf="/org"
    if "${merge_utt}"; then
        train_suf="/merged"
    fi

    for task in "asr" "st"; do
        for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
            # for dset in ${extra_valid_set} ${test_sets}; do
            # for dset in ${train_set}; do
            # for dset in ${valid_set} ${extra_valid_set}; do
            # for dset in ${test_sets}; do
            if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
                _suf="/org"
            elif [ "${dset}" = "${train_set}" ]; then
                _suf="/org"
                dset="${train_set}"
            else
                _suf=""
            fi
            _logdir="${st_exp}/logdir/inference_mtl/${task}/${src_lang}/${train_set}/${dset}/${peft_method}${train_suf}${decode_suf}"
            mkdir -p "${_logdir}"
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
                    -i "${_srcdir}/st.${src_lang}-${tgt_lang}.${dset}.stm" \
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
                        -r /exp/scale23/data/3-way/${src_lang}/st.${src_lang}-${tgt_lang}.${dset}.stm
                fi
            fi

            _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mtl/${task}/${peft_method}${train_suf}${decode_suf}"
            _modeldir="${st_exp}/${src_lang}/${train_set}/mtl/${peft_method}"

            if [ "${dset}" = "${train_set}" ]; then
                ${python} pyscripts/utils/filter_sp.py \
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

            # 2. Submit jobs
            log "Inference started... log: '${_logdir}/decode.*.log'"

            opts=
            if [ "${framework}" == "huggingface" ]; then
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

                inference_tool="pyscripts/utils/hf_whisper_inference.py"
            else
                inference_tool="pyscripts/utils/whisper_inference.py"
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
                ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06' --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
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

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Run evaluation on the MTL decoded data."

    decode_suf="_org"
    if "${merge_decode}"; then
        decode_suf="_merged"
    fi
    train_suf="/org"
    if "${merge_utt}"; then
        train_suf="/merged"
    fi

    for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
        # for dset in ${valid_set}; do
        # for dset in ${test_sets}; do
        # for dset in ${extra_valid_set} ${test_sets}; do
        log "Running ASR evaluation on ${dset}"
        eval_script=run-asr-eval.sh

        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mtl/asr/${peft_method}${train_suf}${decode_suf}"
        _asr_hyp="${PWD}/${_dir}/text"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts=
        if [ "${src_lang}" == "ara" ]; then
            opts+=" --arabic true "
        fi
        opts+=" --cer ${eval_cer} "

        if "${merge_decode}"; then
            opts+=" --merge_utt true "
            opts+=" --data_base_dir ${merged_data_base} "
        fi

        cd evaluation
        ${eval_script} \
            --src_lang ${src_lang} \
            --hyp_asr "${_asr_hyp}" \
            --sclite ${sclite_path} \
            --dset "${_dset}" \
            --score_dir scores_ft/mtl/asr/hf_whisper_${model_name}/${src_lang}/${peft_method}/${train_set}${train_suf}${decode_suf}/${dset} \
            --framework "${framework}" ${opts}
        cd -
    done

    # Note that we assume the evaluation code is available in the path
    for dset in ${valid_set} ${extra_valid_set} ${test_sets}; do
        # for dset in ${valid_set}; do
        # for dset in ${test_sets}; do
        # for dset in ${extra_valid_set} ${test_sets}; do
        # for dset in ${extra_valid_set}; do
        log "Running ST evaluation on ${dset}"
        if [ "${dset}" = "${valid_set}" ] || [ "${dset}" = "${extra_valid_set}" ]; then
            eval_script=run-devset-eval.sh
        elif [ "${dset}" = "fleurs_test" ]; then
            eval_script=run-ood-eval.sh
        else
            eval_script=run-testset-eval.sh
        fi

        _dir="${st_exp}/${src_lang}/decode/${train_set}/${dset}/mtl/st/${peft_method}${train_suf}${decode_suf}"
        _st_hyp="${PWD}/${_dir}/text"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts=
        if [ "${src_lang}" == "ara" ]; then
            opts+=" --arabic true "
        fi

        if "${merge_decode}"; then
            opts+=" --merge_utt true "
            opts+=" --data_base_dir ${merged_data_base} "
        fi

        cd evaluation
        ${eval_script} \
            --src_lang ${src_lang} \
            --hyp_mt "${_st_hyp}" \
            --model_tag ${model_name} \
            --dset "${_dset}" \
            --score_dir scores_ft/mtl/st/hf_whisper_${model_name}/${src_lang}/${peft_method}/${train_set}${train_suf}${decode_suf}/${dset} \
            --framework "${framework}" ${opts}
        cd -
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
