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
run_st=false     # Run ST scoring
run_asr=true     # Run ASR scoring
python=python3
model_tag=base
sclite=/home/hltcoe/cxiao/research/espnet-st/tools/sctk/bin/sclite
dset=dev

declare -A dev_sets_dict
dev_sets_dict+=(["ara"]="dev1 dev2")

# Options
st_hyp_file=                                                                           # Hypothesis file for ST
asr_hyp_file=/home/hltcoe/cxiao/scale23/whisper/recipe/st/exp/st_whisper_base/dev/text # Hypothesis file for ASR
use_cer=false                                                                          # If true use WER, otherwise use CER

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

declare -A cts_testset_dict
declare -A ood_testset_dict

cts_testset_dict+=(["ara"]="iwslt22" ["cmn"]="bbn_cts_bolt" ["kor"]="uhura" ["rus"]="uhura" ["spa"]="fisher")
ood_testset_dict+=(["ara"]="fleurs" ["cmn"]="fleurs" ["kor"]="fleurs" ["rus"]="fleurs" ["spa"]="fleurs")

if [[ ${dset} == *"dev"* ]]; then
    settype="dev"
else
    settype="test"
fi

if [[ "${dset}" == *"${ood_testset_dict[${src_lang}]}"* ]]; then
    dset="${ood_testset_dict[${src_lang}]}"
    stm_dir=/exp/scale23/data/3-way/${src_lang}/testsets/ood
    testset=${ood_testset_dict[${src_lang}]}
elif [[ "${dset}" == *"${cts_testset_dict[${src_lang}]}"* ]]; then
    dset="${cts_testset_dict[${src_lang}]}"
    stm_dir=/exp/scale23/data/3-way/${src_lang}/testsets/cts
    testset=${cts_testset_dict[${src_lang}]}
else
    stm_dir=/exp/scale23/data/3-way/${src_lang}
    testset=${cts_testset_dict[${src_lang}]}
fi

test_score_dir=${score_dir}/asr/${model_tag}_${testset}_${src_lang}_${settype}
mkdir -p "${test_score_dir}/data"

if [ "${settype}" = "dev" ]; then
    # Convert STM files to text and utt2spk files
    cat $stm_dir/sr.${src_lang}-${src_lang}.dev.stm >"${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.dev.stm"
else
    cat $stm_dir/sr.${src_lang}-${src_lang}.${testset}.test.stm >"${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.test.stm"
fi

st_opts=""
asr_opts=""

if "${run_st}"; then
    st_opts="--run_st true \
            --st_ref_file ${test_score_dir}/text.tc.eng \
            --st_utt2spk ${test_score_dir}/utt2spk \
            --st_hyp_file ${st_hyp_file}"
fi

if "${run_asr}"; then
    asr_opts="--run_asr true \
             --asr_ref_file ${test_score_dir}/text.tc.${src_lang} \
             --asr_utt2spk ${test_score_dir}/utt2spk \
             --asr_hyp_file ${asr_hyp_file}"
fi

# Clean the reference file, remove the --- which is some sort of special character for sclite
pyscripts/utils/clean_stm.py \
    -i "${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.${settype}.stm" \
    -o "${test_score_dir}/data/ref.stm"

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm.py \
    -i "${asr_hyp_file}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "${test_score_dir}/data/ref.stm" \
    --dset ${dset}

pyscripts/utils/clean_stm.py \
    -i "${test_score_dir}/data/_hyp.stm" \
    -o "${test_score_dir}/data/hyp.stm"

# Run the eval script
pyscripts/utils/stm_wer.py \
    ${sclite} "${test_score_dir}/data/ref.stm" "${test_score_dir}/data/hyp.stm" \
    "${test_score_dir}/results"
