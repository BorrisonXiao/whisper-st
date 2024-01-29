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
hyp_file=""
ref_file=""
output_hyp_stm=""
stm_format_ref=true
output_ref_stm=""
working_dir=tmp

help_message=$(cat << EOF
Usage: $0 --hyp_file <path to hyp_file> --ref_file <path to ref_file> --stm_format_ref <true or false>

Options:
    hyp_file        # Hypothesis file to convert to STM.
    ref_file        # Reference file to use STM columns from or to convert to STM.
    output_hyp_stm  # Specify name for output hyp STM file.
    stm_format_ref  # Flag to describe ref file format. If true, will use STM information to add to hyp_file (default="${stm_format_ref}").
    output_ref_stm  # Specify name for output ref STM file. Required if --stm_format_ref false.
    working_dir     # Working directory to store intermediate files (default="${working_dir}").
EOF
)

log "$0 $*"

run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ ! -d $working_dir ]; then
    mkdir -p $working_dir
fi

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ${stm_format_ref} == true ]; then
    echo "Get STM column information from $ref_file and create $output_hyp_stm"
    cut -d' ' -f1-6 $ref_file > $working_dir/stm_cols.txt

    paste -d ' ' \
        $working_dir/stm_cols.txt \
        $hyp_file \
        > $output_hyp_stm
else
    if [ "$output_ref_stm" == "" ]; then
        echo "--output_ref_stm must be set if stm_format_ref false"
        exit 1
    fi
    if [ -f $working_dir/dummy_cols.txt ]; then
        rm $working_dir/dummy_cols.txt
    fi
    echo "Generate dummy values for bitext STM"

    total_lines=$(cat $ref_file | wc -l)

    old_start=0
    for (( i=1; i<=${total_lines}; i++ )); do
        start=$old_start
        end=$i

        echo "dummy_path A dummy_spk $start $end <O>" >> $working_dir/dummy_cols.txt
        old_start=$end
    done

    echo "Create $output_hyp_stm"
    paste -d ' ' \
        $working_dir/dummy_cols.txt \
        $hyp_file \
        > $output_hyp_stm

    echo "Create $output_ref_stm"
    paste -d ' ' \
        $working_dir/dummy_cols.txt \
        $ref_file \
        > $output_ref_stm
fi