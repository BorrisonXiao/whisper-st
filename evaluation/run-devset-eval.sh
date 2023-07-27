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
python=python3
score_dir=scores # Top directory to store results
model_tag=base   # Place holder for api consistency
merge_utt=false
hyp_mt=
arabic=false
dset=dev
framework=openai
data_base_dir=/exp/scale23/data/3-way

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

declare -A cts_devset_dict

cts_devset_dict+=(["ara"]="iwslt22" ["cmn"]="bbn_cts_bolt" ["kor"]="uhura" ["rus"]="uhura" ["spa"]="fisher+callhome")
devset=${cts_devset_dict[${src_lang}]}

_prefix=
if [ "${framework}" == "huggingface" ]; then
    _prefix+="hf_"
fi

stm_dir=${data_base_dir}/${src_lang}

test_score_dir=${score_dir}/${devset}
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
    -r "$stm_dir/st.${src_lang}-eng.${dset}.stm" \
    --dset ${dset} ${_opts}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_mt "${test_score_dir}/data/_hyp.stm" \
    --ref_mt "$stm_dir/st.${src_lang}-eng.${dset}.stm" \
    --arabic "${arabic}" \
    --python "${python}"

# Convert STM files to text and utt2spk files
# for stmfile in "$stm_dir/sr.${src_lang}-${src_lang}.${devset}.dev"*".stm"; do
#     set=$(echo $stmfile | awk -F'[_.]' '{print $(NF-1)}')
#     python pyscripts/utils/convert_stm.py $stmfile ${test_score_dir} ${set}.text.tc.${src_lang}
#     python pyscripts/utils/convert_stm.py "$stm_dir/st.${src_lang}-eng.${devset}.${set}.stm" ${test_score_dir} ${set}.text.tc.eng
# done
# python pyscripts/utils/convert_stm.py "$stm_dir/sr.${src_lang}-${src_lang}.dev.stm" ${test_score_dir} dev.text.tc.${src_lang}
# python pyscripts/utils/convert_stm.py "$stm_dir/st.${src_lang}-eng.dev.stm" ${test_score_dir} dev.text.tc.eng

# cat ${test_score_dir}/*.text.tc.${src_lang} >${test_score_dir}/text.tc.${src_lang}
# cat ${test_score_dir}/*.text.tc.eng >${test_score_dir}/text.tc.eng

# st_opts=""

# if "${run_st}"; then
#     st_opts="--run_st true \
#             --st_ref_file ${test_score_dir}/text.tc.eng \
#             --st_utt2spk ${test_score_dir}/utt2spk \
#             --st_hyp_file ${st_hyp_file}"
# fi

# ./run-general-metrics.sh \
#     --score_dir ${test_score_dir} \
#     ${st_opts} \
#     ${asr_opts}
