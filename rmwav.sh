#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Remove the wav files to save disk space
src_langs="ara cmn kor rus spa"
dset=train-all
_sp="_sp"
dumpdir="/exp/cxiao/scale23/dump_scale23"
type=raw

. utils/parse_options.sh

# If the target set does not contain "_test", add the "org/" prefix
if [[ "${dset}" != *_test ]] && [[ "${type}" == raw ]]; then
    _prefix="org/"
else
    _prefix=""
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

for src_lang in ${src_langs}; do
    log "Removing the wav files in ${dumpdir}/${src_lang}/${type}/${_prefix}${dset}${_sp}/wav..."
    if [[ "${type}" == "raw" ]]; then
        rm -rf ${dumpdir}/${src_lang}/${type}/${_prefix}${dset}${_sp}/wav
    else
        rm -rf ${dumpdir}/${src_lang}/${type}/${_prefix}${dset}${_sp}/format.*/data/*.wav
    fi
done
