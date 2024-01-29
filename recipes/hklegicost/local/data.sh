#!/usr/bin/env bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1
. ./db.sh || exit 1

# general configuration
stage=0
stop_stage=100
train_set="train"
dev_set="dev-asr-0 dev-mt-0"
test_set="test"
fs=16000
min_duration=0.5
max_duration=30
dumpdir=dump
data_base_dir=
SECONDS=0

. utils/parse_options.sh || exit 1

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Create Huggingface style dataset"

    _dumpdir=${dumpdir}/raw

    opts="--fs $fs "
    opts+="--srcdir $data_base_dir "
    opts+="--dumpdir $_dumpdir "
    opts+="--rm-tmp "
    opts+="--make-cuts "

    for set in $dev_set; do
        # for set in $train_set $dev_set $test_set; do
        if [ "${set}" == "$train_set" ]; then
            opts+="--min-duration $min_duration "
            opts+="--max-duration $max_duration "
        fi
        log "Preparing $set"

        output_dir=$(pwd)/${_dumpdir}
        mkdir -p $output_dir

        python3 local/create_hf_dset.py --dset $set $opts
    done
fi

#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
# TODO: Do more preprocessing, e.g. lowercasing, removing punctuation, etc.
#fi
