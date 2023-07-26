#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Remove the wav files to save disk space
src_lang=cmn
dset=train-all
_sp="_sp"
dumpdir="/exp/cxiao/scale23/dump_scale23"
type=raw

. utils/parse_options.sh

# If the target set does not contain "_test", add the "org/" prefix
if [[ "${dset}" != *_test ]]; then
    _prefix="org/"
else
    _prefix=""
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Removing the wav files in ${dumpdir}/${src_lang}/${type}/${_prefix}${dset}${_sp}/wav..."

rm -rf ${dumpdir}/${src_lang}/${type}/${_prefix}${dset}${_sp}/wav