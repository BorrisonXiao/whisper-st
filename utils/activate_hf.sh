#!/usr/bin/env bash
# THIS FILE IS GENERATED BY tools/setup_anaconda.sh
if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
. /home/hltcoe/cxiao/research/espnet-st/tools/miniconda/etc/profile.d/conda.sh && conda deactivate && conda activate hf