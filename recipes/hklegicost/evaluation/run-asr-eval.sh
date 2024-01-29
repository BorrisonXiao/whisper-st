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
src_lang=cmn
score_dir=scores # Top directory to store results
python=python3
cer=false
sclite=sclite
hyp_asr=
arabic=false
dset=
data_base_dir=
stereo=true
framework=huggingface

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

opts=
if "${stereo}"; then
    opts+="--stereo "
fi

_suf2=
if "${cer}"; then
    _suf2+="_cer"
fi
test_score_dir=${score_dir}${_suf2}
mkdir -p "${test_score_dir}/data"

if [ -f "${test_score_dir}/result.lc.rm.txt" ]; then
    rm "${test_score_dir}/result.lc.rm.txt"
fi

# Convert the reference file to STM format
pyscripts/utils/csv2stm.py \
    -i ${data_base_dir}/splits/${dset}.csv \
    -o "${test_score_dir}/data/ref.stm"\
    -d ${data_base_dir} \
    --task asr ${opts}

# Convert the hypothesis file to STM format
pyscripts/utils/text2stm.py \
    -i "${hyp_asr}" \
    -o "${test_score_dir}/data/_hyp.stm" \
    -r "${test_score_dir}/data/ref.stm" \
    --dset ${dset}

# Invoke the updated evaluation script
./run_scale23_evals.sh \
    --score_dir "${test_score_dir}" \
    --src_lang "${src_lang}" \
    --hyp_asr "${test_score_dir}/data/_hyp.stm" \
    --ref_asr "${test_score_dir}/data/ref.stm" \
    --arabic "${arabic}" \
    --python "${python}" \
    --cer "${cer}" \
    --sclite "${sclite}"
