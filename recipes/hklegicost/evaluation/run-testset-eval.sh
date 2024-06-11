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
model_tag=base   # Place holder for api consistency
merge_utt=false
python=python3
hyp_mt=
arabic=false
dset=
framework=openai
data_base_dir=/exp/scale23/data/3-way
tgt_lang=eng

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

declare -A cts_testset_dict

cts_testset_dict+=(
    ["bem"]="bigc.test"
    ["mlt"]="main.dev"
    ["bho"]="panlingua.dev"
    ["mar"]="panlingua.test"
    ["tmh"]="taq.dev"
    ["cmn"]="test"
)

stm_dir=${data_base_dir}
testset=${cts_testset_dict[${src_lang}]}

_prefix=
if [ "${framework}" == "huggingface" ]; then
    _prefix+="hf_"
fi

# Hard coded as ASR eval doesn't use this
test_score_dir=${score_dir}
mkdir -p ${test_score_dir}/data

if "${merge_utt}"; then
    _opts="--merge-utt"
else
    _opts=""
fi

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm.py \
    -i "${hyp_mt}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "$stm_dir/st.${src_lang}-${tgt_lang}.${dset}.stm" \
    --dset ${dset} ${_opts}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_mt "${test_score_dir}/data/_hyp.stm" \
    --ref_mt "$stm_dir/st.${src_lang}-${tgt_lang}.${dset}.stm" \
    --arabic "${arabic}" \
    --python "${python}"