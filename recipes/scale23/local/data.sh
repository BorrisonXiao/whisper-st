#!/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0
stop_stage=100
stereo=false
train_set="train"
dev_set="dev-asr-0 dev-mt-0"
test_set="test"
src_lang=
ignore_segments=false
fs_str=16000
min_duration=0.0
start_at_zero=false
datadir=data
SECONDS=0

. utils/parse_options.sh || exit 1;

data_base_dir=/home/cxiao7/research/legicost/export

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

    mkdir -p $(pwd)/${datadir}

    opts="--fs ${fs_str} --min-duration $min_duration "
    if "${stereo}"; then
        opts+="--stereo "
    fi
    if "${ignore_segments}"; then
        opts+="--ignore-segments "
    fi
    if "${start_at_zero}"; then
        opts+="--start-at-zero "
    fi

    for set in $train_set $dev_set $test_set; do
        log "Preparing $set"

        output_dir=$(pwd)/${datadir}/${set}

        mkdir -p $output_dir

        src_stm=${data_base_dir}/sr.${src_lang}-${src_lang}.${set}.stm
        tgt_stm=${data_base_dir}/st.${src_lang}-eng.${set}.stm

        # First create the source
        python local/convert_stm_to_espnet.py $src_stm $output_dir $opts

        # Create the target within the source
        python local/convert_stm_to_espnet.py $tgt_stm $output_dir/tmp_tgt $opts

        # Copy text from target into $output_dir
        cp $output_dir/tmp_tgt/text $output_dir/text.tc.eng

        # Copy source text to include tc.${src_lang}
        cp $output_dir/text $output_dir/text.tc.${src_lang}

        rm -r ${output_dir}/tmp_tgt
    done
fi

#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # TODO: Do more preprocessing, e.g. lowercasing, removing punctuation, etc.
#fi
