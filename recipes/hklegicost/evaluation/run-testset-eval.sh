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

# General options
src_lang=ara
score_dir=scores # Top directory to store results
python=python3
hyp_mt=
arabic=false
dset=
data_base_dir=
stereo=true
framework=openai

help_message=$(
    cat <<EOF
Usage: $0
EOF
)

log "$0 $*"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

testset="test"

_prefix=
if [ "${framework}" == "huggingface" ]; then
    _prefix+="hf_"
fi

opts=
if "${stereo}"; then
    opts+="--stereo "
fi

# Hard coded as ASR eval doesn't use this
test_score_dir=${score_dir}
mkdir -p ${test_score_dir}/data

# Convert the reference file to STM format
pyscripts/utils/csv2stm.py \
    -i ${data_base_dir}/splits/${dset}.csv \
    -o "${test_score_dir}/data/ref.stm"\
    -d ${data_base_dir} \
    --task mt ${opts}

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm.py \
    -i "${hyp_mt}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "${test_score_dir}/data/ref.stm" \
    --dset ${dset}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_mt "${test_score_dir}/data/_hyp.stm" \
    --ref_mt "${test_score_dir}/data/ref.stm" \
    --arabic "${arabic}" \
    --python "${python}"
