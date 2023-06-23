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
src_lang=
score_dir=    # Top directory to store results
run_st=false  # Run ST scoring
run_asr=false # Run ASR scoring
python=python3
model_tag=base
dset=

# Options
st_hyp_file=  # Hypothesis file for ST
asr_hyp_file= # Hypothesis file for ASR
use_cer=false # If true use WER, otherwise use CER

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

. ./path.sh
. ./cmd.sh

declare -A cts_devset_dict

cts_devset_dict+=(["ara"]="iwslt22" ["cmn"]="bbn_cts_bolt" ["kor"]="uhura" ["rus"]="uhura" ["spa"]="fisher callhome")
devset=${dset}

stm_dir=/exp/scale23/data/3-way/${src_lang}

# Hard coded as ASR eval doesn't use this
test_score_dir=${score_dir}/st/${model_tag}_${devset}_${src_lang}_dev
mkdir -p ${test_score_dir}

# Convert STM files to text and utt2spk files
# for stmfile in "$stm_dir/sr.${src_lang}-${src_lang}.${devset}.dev"*".stm"; do
#     set=$(echo $stmfile | awk -F'[_.]' '{print $(NF-1)}')
#     python pyscripts/utils/convert_stm.py $stmfile ${test_score_dir} ${set}.text.tc.${src_lang}
#     python pyscripts/utils/convert_stm.py "$stm_dir/st.${src_lang}-eng.${devset}.${set}.stm" ${test_score_dir} ${set}.text.tc.eng
# done
python pyscripts/utils/convert_stm.py "$stm_dir/sr.${src_lang}-${src_lang}.dev.stm" ${test_score_dir} dev.text.tc.${src_lang}
python pyscripts/utils/convert_stm.py "$stm_dir/st.${src_lang}-eng.dev.stm" ${test_score_dir} dev.text.tc.eng

cat ${test_score_dir}/*.text.tc.${src_lang} >${test_score_dir}/text.tc.${src_lang}
cat ${test_score_dir}/*.text.tc.eng >${test_score_dir}/text.tc.eng

st_opts=""
asr_opts=""

if "${run_st}"; then
    st_opts="--run_st true \
            --st_ref_file ${test_score_dir}/text.tc.eng \
            --st_utt2spk ${test_score_dir}/utt2spk \
            --st_hyp_file ${st_hyp_file}"
fi

if "${run_asr}"; then
    echo "No longer supports ASR eval"
    exit 1
fi

./run-general-metrics.sh \
    --score_dir ${test_score_dir} \
    ${st_opts} \
    ${asr_opts}
