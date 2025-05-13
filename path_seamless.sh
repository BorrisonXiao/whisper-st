MAIN_ROOT=${ESPNET_ROOT}
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

. utils/activate_seamless.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PYTHONPATH=/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/seamless/bin:$PYTHONPATH
export PATH=/home/hltcoe/cxiao/research/espnet-st/tools/miniconda/envs/seamless/bin:$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# Specify huggingface save directory
export HF_HOME=/exp/cxiao/.cache/huggingface

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/lib64

# NOTE(kamo): Source at the last to overwrite the setting
# NOTE(Cihan): No need to install moses for the whisper finetuning task
# . local/path.sh

# ml load cuda11.3/toolkit/11.3.1-1
ml load cuda12.0/toolkit/12.0.1-1
ml load gcc/7.2.0