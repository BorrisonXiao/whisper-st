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
sclite=/home/hltcoe/cxiao/research/espnet-st/tools/sctk/bin/sclite
ref_asr=
hyp_asr=/home/hltcoe/cxiao/scale23/whisper/recipe/st/exp/st_whisper_base/dev/text
arabic=false
dset=dev

declare -A dev_sets_dict
dev_sets_dict+=(["ara"]="dev1 dev2")                                                                      # If true use WER, otherwise use CER

help_message=$(cat << EOF
Usage: $0 --score_dir <path_to_dir> --ref_mt <path_to_ref_file> --hyp_mt <path_to_hyp_file>

Options:
    --score_dir     # Directory to store results.
    --src_lang      # Source language trigraph.
    --ref_mt        # Reference file for translation. STM format.
    --hyp_mt        # Hypothesis file for translation. STM format.
    --ref_asr       # Reference file for ASR. STM format.
    --hyp_asr       # Hypothesis file for ASR. STM format.
    --arabic        # Choose Arabic normalization for ASR (default="${arabic}").
    --python        # Specify python command (default="${python}").
    --sclite        # Sclite binary (default="${sclite}").
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

cts_testset_dict+=(["ara"]="iwslt22" ["cmn"]="bbn_cts_bolt" ["kor"]="uhura" ["rus"]="uhura" ["spa"]="fisher callhome")
ood_testset_dict+=(["ara"]="fleurs" ["cmn"]="fleurs" ["kor"]="fleurs" ["rus"]="fleurs" ["spa"]="fleurs")

if [[ ${dset} == *"dev"* ]]; then
    settype="dev"
else
    settype="test"
fi

cts_testlist="${cts_testset_dict[${src_lang}]}"

if [[ "${dset}" == *"${ood_testset_dict[${src_lang}]}"* ]]; then
    dset="${ood_testset_dict[${src_lang}]}"
    stm_dir=/exp/scale23/data/3-way/${src_lang}/testsets/ood
    testset=${ood_testset_dict[${src_lang}]}
elif [[ " ${cts_testlist[*]} " =~ " ${dset} " ]]; then
    stm_dir=/exp/scale23/data/3-way/${src_lang}/testsets/cts
    testset=${dset}
else
    stm_dir=/exp/scale23/data/3-way/${src_lang}
    testset=${dset}
fi

test_score_dir=${score_dir}/asr/${model_tag}_${testset}_${src_lang}_${settype}
mkdir -p "${test_score_dir}/data"

if [ -f "${test_score_dir}/result.lc.rm.txt" ]; then
    rm "${test_score_dir}/result.lc.rm.txt"
fi

if [ "${settype}" = "dev" ]; then
    # Convert STM files to text and utt2spk files
    cat $stm_dir/sr.${src_lang}-${src_lang}.dev.stm >"${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.dev.stm"
else
    cat $stm_dir/sr.${src_lang}-${src_lang}.${testset}.test.stm >"${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.test.stm"
fi

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm.py \
    -i "${hyp_asr}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.${settype}.stm" \
    --dset ${dset}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_asr "${test_score_dir}/data/_hyp.stm" \
    --ref_asr "${test_score_dir}/data/sr.${src_lang}-${src_lang}.${testset}.${settype}.stm" \
    --arabic "${arabic}" \
    --python "${python}" \
    --sclite "${sclite}"
