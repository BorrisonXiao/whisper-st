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

# General configuration
datadir=/home/cxiao7/research/legicost/export
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_data_prep=false  # Skip data preparation stages.
nj=32                 # The number of parallel jobs.
dumpdir=dump          # Directory to dump features.
python=python3        # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --nj             # The number of parallel jobs (default="${nj}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation"
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts} --dumpdir ${dumpdir} --data_base_dir ${datadir}
    fi
else
    log "Skip the stages for data preparation"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"