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

# General configuration
datadir=
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_data_prep=false  # Skip data preparation stages.
nj=32                 # The number of parallel jobs.
dumpdir=dump          # Directory to dump features.
python=python3        # Specify python to execute espnet commands.
framework=huggingface # huggingface, openai
hf_datadir=           # Directory to the hugging face dataset.
merge_utt=false       # Whether to merge utterances to the closest 30s for training and inference
merged_data_base=     # Base directory for merged data
remove_ark=false      # Whether to remove ark files after merging
python_hf=python3     # Specify python to execute hugging face commands.
src_lang=es           # source language abbrev. id (e.g., es)
tgt_lang=en           # target language abbrev. id (e.g., en)
use_src_lang=true     # Incorporate ASR loss (use src texts) or not
gaussian_merge=false  # Whether to merge the data s.t. the durations follow Gaussian distribution

# Data preparation related
local_data_opts= # The options given to local/data.sh.
save_wav=false   # Whether to save the generated audio (only in feats_type=raw).

# Tokenization related
src_case=lc.rm
tgt_case=tc

# Speed perturbation related
speed_perturb_factors= # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k            # Sampling rate.

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
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --nj             # The number of parallel jobs (default="${nj}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    
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

# Extra files for translation process
if [ $use_src_lang = true ]; then
    utt_extra_files="text.${src_case}.${src_lang} text.${tgt_case}.${tgt_lang}"
else
    utt_extra_files="text.${tgt_case}.${tgt_lang}"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for ${datadir}/${train_set}, ${datadir}/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        if [ -n "${extra_valid_set}" ]; then
            local_data_opts+=" --extra_dev_set ${extra_valid_set}"
        fi
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
            log "Stage 2: Speed perturbation: $datadir/${train_set} -> $datadir/${train_set}_sp"
            for factor in ${speed_perturb_factors}; do
                if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                    scripts/utils/perturb_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" \
                        "${factor}" "$datadir/${train_set}" "$datadir/${train_set}_sp${factor}"
                    _dirs+="$datadir/${train_set}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    _dirs+="$datadir/${train_set} "
                fi
            done
            utils/combine_data.sh --extra_files "${utt_extra_files}" "$datadir/${train_set}_sp" ${_dirs}
            for extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py $datadir/"${train_set}_sp"/${extra_file} >$datadir/"${train_set}_sp"/${extra_file}.tmp
                mv $datadir/"${train_set}_sp"/${extra_file}.tmp $datadir/"${train_set}_sp"/${extra_file}
            done
        else
            log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        # Switching to the espnet environment for data preparation
        . ./path.sh
        which python
        # Now supports only raw data extraction
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: $datadir/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}" "${extra_valid_set}" ${test_sets}; do
                # for dset in ${extra_valid_set}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print $datadir/"${dset}" "${data_feats}${_suf}/${dset}"

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls $datadir/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                echo "${expand_utt_extra_files}"
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done

                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e $datadir/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments $datadir/${dset}/segments "
                fi
                if "${save_wav}"; then
                    _opts+="--save-wav true "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "$datadir/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi

        if "${remove_ark}"; then
            for dset in "${train_set}" "${valid_set}" "${extra_valid_set}" ${test_sets}; do
                log "Removing ark files for ${dset}..."
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi

                rm -rf "${data_feats}${_suf}/${dset}/data"
            done
        fi
        # Switching back to the hugging face environment
        . ./path_hf.sh
        which python
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Merge the wav.scp for the raw wav files to be decoded."
        for dset in "${train_set}" "${valid_set}" "${extra_valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            log "Creating ${data_feats}${_suf}/${dset}/wav_raw.scp..."
            cat ${data_feats}${_suf}/${dset}/wav/*/wav.scp >${data_feats}${_suf}/${dset}/wav_raw.scp
        done
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Export the data directory and merge the utterances if specified."
        stm_exportdir=${dumpdir}/export
        ${python} pyscripts/audio/export_wav.py \
            --split-dev \
            --src-lang ${src_lang} \
            --outdir ${stm_exportdir} \
            --dumpdir ${dumpdir}

        if "${merge_utt}"; then
            _logdir="${dumpdir}/merge_utt/${src_lang}/logdir"
            mkdir -p "${_logdir}"
            if "${gaussian_merge}"; then
                log "Merge the utterances with Gaussian distribution."
                pyscripts/utils/merge_utts_gaussian.py \
                    --input_base_dir ${stm_exportdir} \
                    --output_base_dir "${_logdir}/tmp" \
                    --src_lang ${src_lang} \
                    --tgt_lang ${tgt_lang} \
                    --num_outputs ${nj} \
                    --splits ${train_set} ${valid_set} ${extra_valid_set} ${test_sets}
            else
                pyscripts/utils/merge_utts.py \
                    --input_base_dir ${stm_exportdir} \
                    --output_base_dir "${_logdir}/tmp" \
                    --src_lang ${src_lang} \
                    --tgt_lang ${tgt_lang} \
                    --num_outputs ${nj} \
                    --splits ${train_set} ${valid_set} ${extra_valid_set} ${test_sets}
            fi

            for _path in "${_logdir}/tmp"/*; do
                dset=${_path##*/}

                # If the merged stm file exists already, don't do it again
                log "Merging utterances for ${dset}"
                # ${python} pyscripts/utils/generate_merged_utts.py \
                #     --keyfile ${_path}/keys.${dset}.1.scp \
                #     --dumpdir ${dumpdir}/merged/${dset}/format.1 \
                #     --output_dir ${merged_data_base}/${src_lang} \
                #     --dump-prefix ""
                ${decode_cmd} JOB=1:"${nj}" "${_logdir}/${dset}"/merge.JOB.log \
                    ${python} pyscripts/utils/generate_merged_utts.py \
                    --keyfile ${_path}/keys.${dset}.JOB.scp \
                    --dumpdir ${dumpdir}/merged/${dset}/format.JOB \
                    --output_dir ${merged_data_base}/${src_lang} \
                    --dump-prefix ""

                # Merge the resulted stm files
                mkdir -p ${merged_data_base}/${src_lang}
                # There is a system lag in the file system, so wait for a while
                sleep 5
                # If dset is a test set, i.e. it contains the "_test" substring, add a suffix to the langdir
                if [[ ${dset} == *"_test" ]]; then
                    _suf="/testsets"
                else
                    _suf=""
                fi
                mkdir -p "${merged_data_base}/${src_lang}${_suf}"
                for i in $(seq "${nj}"); do
                    cat "${dumpdir}/merged/${dset}/format.${i}/merged.sr.stm"
                done >"${merged_data_base}/${src_lang}${_suf}/sr.${src_lang}-${src_lang}.${dset}.stm"
                for i in $(seq "${nj}"); do
                    cat "${dumpdir}/merged/${dset}/format.${i}/merged.st.stm"
                done >"${merged_data_base}/${src_lang}${_suf}/st.${src_lang}-${tgt_lang}.${dset}.stm"
            done
        fi
    fi

    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Convert the data into huggingface datasets"

        stm_exportdir=${dumpdir}/export
        if "${merge_utt}"; then
            src_dir=${merged_data_base}
        else
            src_dir=${stm_exportdir}
        fi
        scripts/utils/create_dataset.sh \
            --python ${python_hf} \
            --src_lang ${src_lang} \
            --raw_data_location ${src_dir} \
            --output_path "${hf_datadir}" \
            --stm "${merge_utt}"
    fi
else
    log "Skip the stages for data preparation"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
