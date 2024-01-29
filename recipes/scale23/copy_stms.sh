#!/usr/bin/env bash

script=/exp/scale23/results/copy_stm.sh
srcdir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_ft/st
mtlstdir=/home/hltcoe/cxiao/scale23/st/evaluation/scores_ft/mtl/st

for stm in ${srcdir}/hf_whisper_{large-v2,medium}/{ara,cmn,kor,rus,spa}/{lora,none}/{train-cts_sp,train-all_sp}/merged_org/{dev1,dev2,fleurs_test,bbn_cts_bolt_test,iwslt22_test,fisher_test,callhome_test,uhura_test}/hyp_mt.stm; do
    # If the file does not exist, skip
    if [ ! -f "${stm}" ]; then
        continue
    fi
    # Skip the fleurs dataset as it's not merged anyways
    if [[ ${stm} == *"fleurs"* ]]; then
        continue
    fi
    _path=${stm%/hyp_mt.stm}
    dset=${_path##*/}
    # Remove the _test suffix in dset
    dset=${dset%_test}
    _path=${_path%/*}
    setting=${_path##*/}
    _path=${_path%/*}
    train_set=${_path##*/}
    _path=${_path%/*}
    peft=${_path##*/}
    _path=${_path%/*}
    lang=${_path##*/}
    _path=${_path%/*}
    model=${_path##*/}

    _name=${model}.${lang}.${peft}.${train_set}.${setting}.${dset}.cihan.stm
    _fullpath=/exp/scale23/results/english-language/${lang}/${dset}/${_name}
    if [ -f "${_fullpath}" ]; then
        echo "${_fullpath} exists, skipping"
        continue
    fi
    echo "Copying ${_name}"

    # Run the script
    ${script} ${stm} ${lang} "eng" ${dset} ${_name}
done

for stm in ${mtlstdir}/hf_whisper_{large-v2,medium}/{ara,cmn,kor,rus,spa}/{lora,none}/{train-cts_sp,train-all_sp}/merged_org/{dev1,dev2,fleurs_test,bbn_cts_bolt_test,iwslt22_test,fisher_test,callhome_test,uhura_test}/hyp_mt.stm; do
    # If the file does not exist, skip
    if [ ! -f "${stm}" ]; then
        continue
    fi
    # Skip the fleurs dataset as it's not merged anyways
    if [[ ${stm} == *"fleurs"* ]]; then
        continue
    fi
    _path=${stm%/hyp_mt.stm}
    dset=${_path##*/}
    # Remove the _test suffix in dset
    dset=${dset%_test}
    _path=${_path%/*}
    setting=${_path##*/}
    _path=${_path%/*}
    train_set=${_path##*/}
    _path=${_path%/*}
    peft=${_path##*/}
    _path=${_path%/*}
    lang=${_path##*/}
    _path=${_path%/*}
    model=${_path##*/}

    _name=${model}.${lang}.mtl.${peft}.${train_set}.${setting}.${dset}.cihan.stm
    _fullpath=/exp/scale23/results/english-language/${lang}/${dset}/${_name}
    if [ -f "${_fullpath}" ]; then
        echo "${_fullpath} exists, skipping"
        continue
    fi
    echo "Copying ${_name}"

    # Run the script
    ${script} ${stm} ${lang} "eng" ${dset} ${_name}
done
