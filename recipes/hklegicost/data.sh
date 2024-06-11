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
datadir=data
stage=0              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
nj=32                # The number of parallel jobs.
srcdir=              # Directory to the source data.
python=python3       # Specify python to execute espnet commands.
hf_datadir=          # Directory to the hugging face dataset.
merge_utt=false      # Whether to merge utterances or not for training and inference
merged_data_base=    # Base directory for merged data
python_hf=python3    # Specify python to execute hugging face commands.
src_lang=cmn         # source language abbrev. id (e.g., es)
tgt_lang=eng         # target language abbrev. id (e.g., en)
gaussian_merge=false # Whether to merge the data s.t. the durations follow Gaussian distribution
stm=true             # Whether to use STM data for training and inference
dumpdir=             # Directory to dump features.
dset=                # The dataset to be created

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors= # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k            # Sampling rate.

# [Task dependent] Set the datadir name created by local/data.sh
train_set= # Name of training set.
valid_set= # Name of validation set used for monitoring/tuning network training.
test_sets= # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --nj             # The number of parallel jobs (default="${nj}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
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

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Convert the raw data into stm files"

    for mode in "st" "sr"; do
        for dset in ${train_set} ${valid_set} ${test_sets}; do
            local/legicost.py \
                --src-lang ${src_lang} \
                --tgt-lang ${tgt_lang} \
                --dset ${dset} \
                --mode ${mode} \
                -i ${srcdir} \
                -o ${datadir}/raw
        done
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Create segments based on the stm files to form the new stm files"

    reseg_dir=${datadir}/resegmented
    audio_dir=${dumpdir}/wav
    mkdir -p "${audio_dir}"

    for mode in "st" "sr"; do
        _mode=${mode}
        _tgt_lang=${tgt_lang}
        if [ "${mode}" == "sr" ]; then
            _mode=asr
            _tgt_lang=${src_lang}
        fi

        for dset in ${train_set} ${valid_set} ${test_sets}; do
            _input="${datadir}/raw/${_mode}/${mode}.${src_lang}-${_tgt_lang}.${dset}.stm"
            _output="${reseg_dir}/${mode}.${src_lang}-${_tgt_lang}.${dset}.stm"
            pyscripts/utils/resegment_audio.py \
                -i "${_input}/" \
                -o "${_output}" \
                --audio-dir "${audio_dir}"
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Convert the data into huggingface datasets"
    opts=
    if [ -n "${dset}" ]; then
        opts+=" --dset ${dset} "
    fi
    scripts/utils/create_dataset.sh \
        --python ${python_hf} \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --raw_data_location ${datadir}/resegmented \
        --output_path "${hf_datadir}" \
        --stm "${stm}" ${opts}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
