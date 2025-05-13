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
src_lang=fr
score_dir=scores_ft_covost2 # Top directory to store results
python=python3
hyp_mt=/exp/cxiao/scale23/hf_data_covost2/fr/test.text
dset=test
data_base_dir=/exp/cxiao/scale23/hf_data_covost2/fr
no_glm=true

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

declare -A testset_dict

testset_dict+=(["fr"]="test" ["de"]="test")
testset=${testset_dict[${src_lang}]}

# Hard coded as ASR eval doesn't use this
test_score_dir=${score_dir}
mkdir -p ${test_score_dir}/data

cat $data_base_dir/${testset}.tgt.stm >"${test_score_dir}/data/${testset}.tgt.stm"

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm_covost2.py \
    -i "${hyp_mt}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "$data_base_dir/${testset}.tgt.stm" \
    --dset ${dset}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_mt "${test_score_dir}/data/_hyp.stm" \
    --ref_mt "${test_score_dir}/data/${testset}.tgt.stm" \
    --no_glm "${no_glm}" \
    --python "${python}"
