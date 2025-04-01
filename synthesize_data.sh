#!/usr/bin/env bash

#$ -cwd

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# This script is used to run the final mult-task, i.e. ST + ASR + MT + Prompted-ST, experiments
# Note that this version uses masked GT prompt at training time

# Change the following according to your experiments
src_lang=spa
tgt_lang=eng

src_dset=dev2
tgt_dset=dev3
key_file=
stage=1
stop_stage=2
model_name=large-v2
hf_datadir=/exp/cxiao/scale23/hf_data
save_dir=
dumpdir=
inference_nj=8
inference_batch_size=16

debug=false
# debug=true

# Modify this to your python path, this is due to some ESPNet environment issues
python_hf=python3

. ./path_hf.sh
. ./cmd.sh

. ./utils/parse_options.sh

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

# local/prep_covost2.py \
#     --data-dir ${dumpdir}/raw \
#     --save-dir ${hf_datadir}/${src_lang}

# Run Whisper inference on the validation data to create the ASR-prompted validation2 data
logdir="${dumpdir}/log/${src_dset}"
mkdir -p "${logdir}"
output_dir="${dumpdir}/decode/${src_dset}"
mkdir -p "${output_dir}"

# 1. Split the key file
_nj=$(min "${inference_nj}" "$(wc <${key_file} -l)")

split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${logdir}/decode.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}

# opts=" --dset ${hf_datadir}/${src_lang}/${src_dset} "
# inference_tool="pyscripts/utils/hf_whisper_inference.py"

# if "${debug}"; then
#     ${inference_tool} \
#         --keyfile ${logdir}/decode.1.scp \
#         --src-lang ${src_lang} \
#         --tgt-lang ${src_lang} \
#         --output_dir ${logdir}/output.1 \
#         --batch-size ${inference_batch_size} \
#         --model_name large-v2 \
#         --num-beams 1 ${opts}
# else
#     # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
#     #       but it's used only for deciding the sample ids.
#     # shellcheck disable=SC2046,SC2086
#     ${cuda_cmd} --hostname '!r5n0*\&!r10n04\&!r10n06\&!r7n01' --mem 16G --gpu 1 JOB=1:"${_nj}" "${logdir}"/decode.JOB.log \
#         ${inference_tool} \
#         --keyfile ${logdir}/decode.JOB.scp \
#         --src-lang ${src_lang} \
#         --tgt-lang ${src_lang} \
#         --output_dir ${logdir}/output.JOB \
#         --batch-size ${inference_batch_size} \
#         --model_name large-v2 \
#         --num-beams 1 ${opts}
# fi

# # 3. Concatenates the output files from each jobs
# mkdir -p "${output_dir}"
# for i in $(seq "${_nj}"); do
#     cat "${logdir}/output.${i}/text"
# done | LC_ALL=C sort -k1 >"${output_dir}/text"

# # Create the synthesized data
# pyscripts/utils/create_synth_data.py \
#     --src-dset ${hf_datadir}/${src_lang}/${data_set} \
#     --tgt-dset ${save_dir} \
#     --asr-hyp ${output_dir}/text
