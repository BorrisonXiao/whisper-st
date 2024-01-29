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

input_file= # File to clean.
utt2spk= # File containing utt2spk information.
input_type=ref # ref or hyp.
score_dir=  # Directory to store scores.
nlsyms_txt=none # Non-linguistic symbol list if existing.
use_cleaner=false # Use cleaner.
cleaner=none     # Text cleaner.
file_idx=     # Index of the file (for multi-reference).
python=python3

help_message=$(cat << EOF
Usage: $0 --input_file "<path_to_file>" --utt2spk "<path_to_file>" --score_dir "<path_to_dir>"

Options:
    --input_file    # File to clean.
    --utt2spk       # File containing utt2spk information.
    --input_type    # ref or hyp (default="${input_type}).
    --score_dir     # Directory to store scores.
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --use_cleaner   # Use cleaner.
    --cleaner       # Text cleaner (default="${cleaner}").
    --file_idx      # Index of the file (for multi-reference).
EOF
)

log "$0 $*"
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

. ./path.sh

if "${use_cleaner}"; then
    cleaner_opts="--cleaner ${cleaner}"
else
    cleaner_opts=""
fi

idx_opt=
if [ "${file_idx}" != "" ]; then
    idx_opt+=".${file_idx}"
fi

echo "Reformat input text"

echo "Sorting file"

sort -t ' ' -k1,1 -k2,2r ${input_file} > ${score_dir}/${input_type}.sorted${idx_opt}

paste \
    <(<"${score_dir}/${input_type}.sorted${idx_opt}" \
        ${python} -m espnet2.bin.tokenize_text  \
            -f 2- --input - --output - \
            --token_type word \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --remove_non_linguistic_symbols true \
            ${cleaner_opts} \
            ) \
    <(<"${utt2spk}" awk '{ print "(" $2 "-" $1 ")" }') \
        >"${score_dir}/${input_type}.trn.org${idx_opt}"

echo "Remove utterance id"

perl -pe 's/\([^\)]+\)$//g;' "${score_dir}/${input_type}.trn.org${idx_opt}" > "${score_dir}/${input_type}.trn${idx_opt}"

echo "Run detokenizer"

detokenizer.perl -l en -q < "${score_dir}/${input_type}.trn${idx_opt}" > "${score_dir}/${input_type}.trn.detok${idx_opt}"

remove_punctuation.pl < "${score_dir}/${input_type}.trn.detok${idx_opt}" > "${score_dir}/${input_type}.trn.detok.lc.rm${idx_opt}"

# TODO: Run any other cleaning steps we need (e.g. lowercasing, removing punctuations, etc.)

echo "Successfully finished cleaning."
