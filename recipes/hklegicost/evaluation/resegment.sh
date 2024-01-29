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

src_lang=cmn
tgt_lang=eng
model_name=medium
stdir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_ft/st
refdir=/exp/scale23/data/3-way
resegdir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_reseg/st
mode=st
python=python3
stage=4
stop_stage=4

. utils/parse_options.sh

SECONDS=0

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Converting the references to xml files"
    ${python} pyscripts/utils/stm2xml.py \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --refdir ${refdir} \
        --outdir ${resegdir}/${src_lang}/xml
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Converting the hypothesis stm files to txt files"
    ${python} pyscripts/utils/stm2txt.py \
        --input_dir ${stdir}/hf_whisper_${model_name}/${src_lang} \
        --output_dir ${resegdir}/${src_lang}/txt/${model_name}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Performing the MWER segmentation"
    for hypmt in ${resegdir}/${src_lang}/txt/${model_name}/{lora,none}/{train-cts_sp,train-all_sp}/merged_merged/*/hyp_mt.txt; do
        # If the file does not exist, skip
        if [ ! -f "${hypmt}" ]; then
            continue
        fi
        # Skip the fleurs dataset as it's not merged anyways
        if [[ ${hypmt} == *"fleurs"* ]]; then
            continue
        fi
        _path=${hypmt%/hyp_mt.txt}
        dset=${_path##*/}
        log "Processing ${dset}"
        _path=${_path%/*}
        _path=${_path%/*}
        train_set=${_path##*/}
        _path=${_path%/*}
        peft=${_path##*/}
        srxml=${resegdir}/${src_lang}/xml/${dset}/ref_sr.xml
        mtxml=${resegdir}/${src_lang}/xml/${dset}/ref_mt.xml
        outdir=${resegdir}/${src_lang}/aligned/${model_name}/${peft}/${train_set}/${dset}
        mkdir -p ${outdir}

        # Perform the MWER segmentation
        ./mwerSegment/segmentBasedOnMWER.sh \
        ${srxml} \
        ${mtxml} \
        ${hypmt} \
        whipser \
        English \
        ${outdir}/aligned.xml \
        no_normalize \
        1
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Converting the aligned xml files to stm files"
    ${python} pyscripts/utils/xml2stm.py \
        --input-dir ${resegdir}/${src_lang}/aligned/${model_name} \
        --output-dir ${resegdir}/${src_lang}/stm/${model_name} \
        --refdir ${refdir}/${src_lang} \
        --mode ${mode}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
