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
model_tag=base
hyp_mt=
arabic=false
dset=dev
framework=openai

help_message=$(
    cat <<EOF
Usage: $0
EOF
)

log "$0 $*"

run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

declare -A ood_testset_dict

# For now only supports single testset
ood_testset_dict+=(["ara"]="fleurs" ["cmn"]="fleurs" ["kor"]="fleurs" ["rus"]="fleurs" ["spa"]="fleurs")

stm_dir=/exp/scale23/data/3-way/${src_lang}/testsets/ood
testset=${ood_testset_dict[${src_lang}]}

_prefix=
if [ "${framework}" == "huggingface" ]; then
    _prefix+="hf_"
fi

# Hard coded as ASR eval doesn't use this
test_score_dir=${score_dir}/st/${_prefix}${model_tag}_${testset}_${src_lang}_test
mkdir -p ${test_score_dir}/data

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm.py \
    -i "${hyp_mt}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "$stm_dir/st.${src_lang}-eng.${dset}.test.stm" \
    --dset ${dset}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_mt "${test_score_dir}/data/_hyp.stm" \
    --ref_mt "$stm_dir/st.${src_lang}-eng.${dset}.test.stm" \
    --arabic "${arabic}" \
    --python "${python}"

# # Convert STM files to text and utt2spk files
# python pyscripts/utils/convert_stm.py $stm_dir/sr.${src_lang}-${src_lang}.${testset}.test.stm ${test_score_dir} text.tc.${src_lang}
# python pyscripts/utils/convert_stm.py $stm_dir/st.${src_lang}-eng.${testset}.test.stm ${test_score_dir} text.tc.eng

# st_opts=""
# asr_opts=""

# if "${run_st}"; then
#     st_opts="--run_st true \
#             --st_ref_file ${test_score_dir}/text.tc.eng \
#             --st_utt2spk ${test_score_dir}/utt2spk \
#             --st_hyp_file ${st_hyp_file}"
# fi

# if "${run_asr}"; then
#     echo "No longer supports ASR eval"
#     exit 1
# fi

# ./run-general-metrics.sh \
#     --score_dir ${test_score_dir} \
#     ${st_opts} \
#     ${asr_opts}
