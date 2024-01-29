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

asrdir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_ft/asr
exportdir=/exp/cxiao/scale23/export_asr

for _dir in {${asrdir}/*/*/*/*/*/*/ref_asr.stm,${asrdir}/*/*/*/*/*/*/hyp_asr.stm}; do
    dir=${_dir%/*}
    dset=${dir##*/}
    dir=${dir%/*}
    setting=${dir##*/}
    dir=${dir%/*}
    train_set=${dir##*/}
    dir=${dir%/*}
    peft_method=${dir##*/}
    dir=${dir%/*}
    src_lang=${dir##*/}
    dir=${dir%/*}
    model=${dir##*/}
    if [[ ${dset} == *"_cer" ]]; then
        continue
    fi
    mkdir -p "${exportdir}/${model}/${src_lang}/${peft_method}/${train_set}/${setting}/${dset}"
    cp "${_dir}" "${exportdir}/${model}/${src_lang}/${peft_method}/${train_set}/${setting}/${dset}"
done
