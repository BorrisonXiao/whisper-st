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

# Evaluation related
sclite_path=/home/hltcoe/cxiao/research/espnet-st/tools/sctk/bin/sclite

# General configuration
datadir=
token_listdir=
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
skip_data_prep=false  # Skip data preparation stages.
skip_train=false      # Skip training stages.
skip_eval=false       # Skip decoding and evaluation stages.
skip_upload=true      # Skip packing and uploading stages.
skip_upload_hf=true   # Skip uploading to hugging face stages.
ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1           # The number of nodes.
nj=32                 # The number of parallel jobs.
inference_nj=32       # The number of parallel jobs in decoding.
gpu_inference=false   # Whether to perform gpu decoding.
dumpdir=dump          # Directory to dump features.
expdir=exp            # Directory to save experiments.
python=python3        # Specify python to execute espnet commands.
model_name=base       # Model name, e.g. "base", "large", etc.
framework=huggingface # huggingface, openai
hf_datadir=           # Directory to the hugging face dataset.

# Data preparation related
local_data_opts= # The options given to local/data.sh.
save_wav=false   # Whether to save the generated audio (only in feats_type=raw).

# Speed perturbation related
speed_perturb_factors= # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
token_joint=false   # whether to use a single bpe system for both source and target languages
src_case=lc.rm
src_token_type=bpe                    # Tokenization type (char or bpe) for source languages.
src_nbpe=30                           # The number of BPE vocabulary for source language.
src_bpemode=unigram                   # Mode of BPE for source language (unigram or bpe).
src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language.
src_bpe_nlsyms=                       # non-linguistic symbols list, separated by a comma, for BPE of source language
src_bpe_char_cover=1.0                # character coverage when modeling BPE for source language
tgt_case=tc
tgt_token_type=bpe                    # Tokenization type (char or bpe) for target language.
tgt_nbpe=30                           # The number of BPE vocabulary for target language.
tgt_bpemode=unigram                   # Mode of BPE (unigram or bpe) for target language.
tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language.
tgt_bpe_nlsyms=                       # non-linguistic symbols list, separated by a comma, for BPE for target language.
tgt_bpe_char_cover=1.0                # character coverage when modeling BPE for target language.
hugging_face_model_name_or_path=""    # Hugging Face model or path for hugging_face tokenizer

# Ngram model related
use_ngram=false
ngram_exp=
ngram_num=3
use_src_ngram=false

# Language model related
use_lm=false     # Use language model for ST decoding.
use_src_lm=false # Use language model for ASR multi-decoder decoding.
lm_tag=          # Suffix to the result dir for language model training.
lm_exp=          # Specify the directory path for LM experiment.
# If this option is specified, lm_tag is ignored.
src_lm_exp=   # Specify the directory path for LM experiment.
lm_stats_dir= # Specify the directory path for LM statistics.
lm_config=    # Config for language model training.
lm_args=      # Arguments for language model training, e.g., "--max_epoch 10".
# Note that it will overwrite args in lm config.
use_word_lm=false     # Whether to use word language model.
use_src_word_lm=false # Whether to use word language model.
num_splits_lm=1       # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ST model related
st_tag= # Suffix to the result dir for st model training.
st_exp= # Specify the directory path for ST experiment.
# If this option is specified, st_tag is ignored.
st_stats_dir= # Specify the directory path for ST statistics.
st_config=    # Config for st model training.
st_args=      # Arguments for st model training, e.g., "--max_epoch 10".
# Note that it will overwrite args in st config.
pretrained_asr=            # Pretrained model to load
ignore_init_mismatch=false # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_st=1            # Number of splitting for lm corpus.
src_lang=es                # source language abbrev. id (e.g., es)
tgt_lang=en                # target language abbrev. id (e.g., en)
use_src_lang=true          # Incorporate ASR loss (use src texts) or not

# Upload model related
hf_repo=

# Decoding related
use_k2=false        # Whether to use k2 based decoder
use_streaming=false # Whether to use streaming decoding
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
# Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth     # Language model path for decoding.
inference_asr_lm=valid.loss.ave.pth # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
# inference_st_model=valid.acc.ave.pth # ST model path for decoding.
# e.g.
# inference_st_model=train.loss.best.pth
# inference_st_model=3epoch.pth
inference_st_model=valid.acc.best.pth
# inference_st_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=                # Name of training set.
valid_set=                # Name of validation set used for monitoring/tuning network training.
test_sets=                # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
src_bpe_train_text=       # Text file path of bpe training set for source language.
tgt_bpe_train_text=       # Text file path of bpe training set for target language.
lm_train_text=            # Text file path of language model training set.
lm_dev_text=              # Text file path of language model development set.
lm_test_text=             # Text file path of language model evaluation set.
nlsyms_txt=none           # Non-linguistic symbol list if existing.
cleaner=none              # Text cleaner.
g2p=none                  # g2p method (needed if token_type=phn).
score_opts=               # The options given to sclite scoring
local_score_opts=         # The options given to local/score.sh.
st_speech_fold_length=800 # fold_length for speech data during ST training.
st_text_fold_length=150   # fold_length for text data during ST training.
lm_fold_length=150        # fold_length for LM training.

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --token_joint=false       # Whether to use a single bpe system for both source and target languages.
                              # if set as true, will use tgt_* for processing (default="${token_joint}").
    --src_token_type=bpe      # Tokenization type (char or bpe) for source languages. (default="${src_token_type}").
    --src_nbpe=30             # The number of BPE vocabulary for source language. (default="${src_nbpe}").
    --src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe). (default="${src_bpemode}").
    --src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language. (default="${src_bpe_input_sentence_size}").
    --src_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE of source language. (default="${src_bpe_nlsyms}").
    --src_bpe_char_cover=1.0  # Character coverage when modeling BPE for source language. (default="${src_bpe_char_cover}").
    --tgt_token_type=bpe      # Tokenization type (char or bpe) for target language. (default="${tgt_token_type}").
    --tgt_nbpe=30             # The number of BPE vocabulary for target language. (default="${tgt_nbpe}").
    --tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language. (default="${tgt_bpemode}").
    --tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language. (default="${tgt_bpe_input_sentence_size}").
    --tgt_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE for target language. (default="${tgt_bpe_nlsyms}").
    --tgt_bpe_char_cover=1.0  # Character coverage when modeling BPE for target language. (default="${tgt_bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the directory path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the directory path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # ST model related
    --st_tag           # Suffix to the result dir for st model training (default="${st_tag}").
    --st_exp           # Specify the directory path for ST experiment.
                       # If this option is specified, st_tag is ignored (default="${st_exp}").
    --st_stats_dir     # Specify the directory path for ST statistics (default="${st_stats_dir}").
    --st_config        # Config for st model training (default="${st_config}").
    --st_args          # Arguments for st model training (default="${st_args}").
                       # e.g., --st_args "--max_epoch 10"
                       # Note that it will overwrite args in st config.
    --pretrained_asr=          # Pretrained model to load (default="${pretrained_asr}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type. (default="${feats_normalize}").
    --num_splits_st    # Number of splitting for lm corpus.  (default="${num_splits_st}").
    --src_lang=        # source language abbrev. id (e.g., es). (default="${src_lang}")
    --tgt_lang=        # target language abbrev. id (e.g., en). (default="${tgt_lang}")
    --use_src_lang=    # Incorporate ASR loss (use src texts) or not 

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_st_model # ST model path for decoding (default="${inference_st_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").
    
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --src_bpe_train_text # Text file path of bpe training set for source language.
    --tgt_bpe_train_text # Text file path of bpe training set for target language
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --st_speech_fold_length # fold_length for speech data during ST training (default="${st_speech_fold_length}").
    --st_text_fold_length   # fold_length for text data during ST training (default="${st_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
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

if [ "${framework}" == "huggingface" ]; then
    . ./path_hf.sh
else
    . ./path.sh
fi
. ./cmd.sh

# Check required arguments
[ -z "${train_set}" ] && {
    log "${help_message}"
    log "Error: --train_set is required"
    exit 2
}
[ -z "${valid_set}" ] && {
    log "${help_message}"
    log "Error: --valid_set is required"
    exit 2
}
[ -z "${test_sets}" ] && {
    log "${help_message}"
    log "Error: --test_sets is required"
    exit 2
}

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for translation process
if [ $use_src_lang = true ]; then
    utt_extra_files="text.${src_case}.${src_lang} text.${tgt_case}.${tgt_lang}"
else
    utt_extra_files="text.${tgt_case}.${tgt_lang}"
fi

# Use the same text as ST for bpe training if not specified.
[ -z "${src_bpe_train_text}" ] && [ $use_src_lang = true ] && src_bpe_train_text="${data_feats}/${train_set}/text.${src_case}.${src_lang}"
[ -z "${tgt_bpe_train_text}" ] && tgt_bpe_train_text="${data_feats}/${train_set}/text.${tgt_case}.${tgt_lang}"
# Use the same text as ST for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text.${tgt_case}.${tgt_lang}"
# Use the same text as ST for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text.${tgt_case}.${tgt_lang}"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text.${tgt_case}.${tgt_lang}"

if [ -z "${token_listdir}" ]; then
    # Check tokenization type
    token_listdir=$datadir/${src_lang}_${tgt_lang}_token_list
    # The tgt bpedir is set for all cases when using bpe
    tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}"
    tgt_bpeprefix="${tgt_bpedir}"/bpe
    tgt_bpemodel="${tgt_bpeprefix}".model
    tgt_bpetoken_list="${tgt_bpedir}"/tokens.txt
    tgt_chartoken_list="${token_listdir}"/char/tgt_tokens.txt
    hugging_face_token_list="${token_listdir}/hugging_face_"${hugging_face_model_name_or_path/\//-}/tokens.txt
else
    token_listdir="${token_listdir}"
    tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}"
    tgt_bpeprefix="${tgt_bpedir}"/bpe
    tgt_bpemodel="${tgt_bpeprefix}".model
    tgt_bpetoken_list="${tgt_bpedir}"/tokens.txt
    tgt_chartoken_list="${token_listdir}"/char/tgt_tokens.txt
    hugging_face_token_list="${token_listdir}/hugging_face_"${hugging_face_model_name_or_path/\//-}/tokens.txt

fi
if "${token_joint}"; then
    # if token_joint, the bpe training will use both src_lang and tgt_lang to train a single bpe model
    src_bpedir="${tgt_bpedir}"
    src_bpeprefix="${tgt_bpeprefix}"
    src_bpemodel="${tgt_bpemodel}"
    src_bpetoken_list="${tgt_bpetoken_list}"
    src_chartoken_list="${tgt_chartoken_list}"
else
    #src_bpedir="${token_listdir}/src_bpe_${src_bpemode}${src_nbpe}"
    src_bpedir=/home/cxiao7/research/espnet-st/egs2/iwslt22_dialect/st_mbart/pretrained/alt-arabic/speech/amir/competitions/IWSLT/MGB2_8KHz2/data/token_list/bpe_unigram2000
    # src_bpedir=/alt-arabic/speech/amir/competitions/IWSLT/MGB2_8KHz2/data/token_list/bpe_unigram2000
    #src_bpeprefix="${src_bpedir}"/bpe
    src_bpeprefix=/home/cxiao7/research/espnet-st/egs2/iwslt22_dialect/st_mbart/pretrained/alt-arabic/speech/amir/competitions/IWSLT/MGB2_8KHz2/data/token_list/bpe_unigram2000/bpe
    src_bpemodel="${src_bpeprefix}".model
    src_bpetoken_list="${src_bpedir}"/tokens.txt
fi
# Set token types for src and tgt langs
if [ $use_src_lang = false ]; then
    src_token_type=none
    src_token_list=none
elif [ "${src_token_type}" = bpe ]; then
    src_token_list="${src_bpetoken_list}"
elif [ "${src_token_type}" = char ]; then
    src_token_list="${src_chartoken_list}"
    src_bpemodel=none
elif [ "${src_token_type}" = word ]; then
    src_token_list="${src_wordtoken_list}"
    src_bpemodel=none
else
    log "Error: not supported --src_token_type '${src_token_type}'"
    exit 2
fi
if [ "${tgt_token_type}" = bpe ]; then
    tgt_token_list="${tgt_bpetoken_list}"
elif [ "${tgt_token_type}" = char ]; then
    tgt_token_list="${tgt_chartoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = word ]; then
    tgt_token_list="${tgt_wordtoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = hugging_face ]; then
    tgt_token_list="${hugging_face_token_list}"
    tgt_bpemodel=${hugging_face_model_name_or_path}
elif [ "${tgt_token_type}" = fairseq ]; then
    log "tgt_token_type: ${tgt_token_type}"
    tgt_token_list="/home/cxiao7/research/iwslt2023/dialect/mbart/tokens.txt"
    tgt_bpemodel="/home/cxiao7/research/iwslt2023/dialect/mbart/sentence.bpe.model"
else
    log "Error: not supported --tgt_token_type '${tgt_token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${tgt_wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${tgt_token_list}"
    lm_token_type="${tgt_token_type}"
fi

# Set tag for naming of model directory
if [ -z "${st_tag}" ]; then
    if [ -n "${st_config}" ]; then
        st_tag="$(basename "${st_config}" .yaml)_${feats_type}"
    else
        st_tag="train_${feats_type}"
    fi
    st_tag+="_${src_lang}_${tgt_lang}_${tgt_token_type}_${tgt_case}"
    if [ "${tgt_token_type}" = bpe ]; then
        st_tag+="${tgt_nbpe}"
    fi
    if [ "${tgt_token_type}" = hugging_face ]; then
        st_tag+="_"${hugging_face_model_name_or_path/\//-}
    fi
    # Add overwritten arg's info
    if [ -n "${st_args}" ]; then
        st_tag+="$(echo "${st_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_tag+="_sp"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)"
    else
        lm_tag="train"
    fi
    lm_tag+="_${src_lang}_${tgt_lang}_${lm_token_type}"
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${tgt_nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${st_stats_dir}" ]; then
    st_stats_dir="${expdir}/st_stats_${feats_type}_${src_lang}_${tgt_lang}_${tgt_token_type}"
    if [ "${tgt_token_type}" = bpe ]; then
        st_stats_dir+="${tgt_nbpe}"
    fi
    if [ "${tgt_token_type}" = hugging_face ]; then
        st_stats_dir+="_"${hugging_face_model_name_or_path/\//-}
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_stats_dir+="_sp"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    lm_stats_dir="${expdir}/lm_stats_${src_lang}_${tgt_lang}_${lm_token_type}"
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${tgt_nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${st_exp}" ]; then
    if [ "${framework}" = "huggingface" ]; then
        st_exp="${expdir}/st_hf_${st_tag}"
    else
        st_exp="${expdir}/st_${st_tag}"
    fi
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi
if [ -z "${ngram_exp}" ]; then
    ngram_exp="${expdir}/ngram"
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    if "${use_ngram}"; then
        inference_tag+="_ngram_$(basename "${ngram_exp}")_$(echo "${inference_ngram}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
        inference_tag+="_use_k2"
    fi
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for ${datadir}/${train_set}, ${datadir}/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
            log "Stage 2: Speed perturbation: $datadir/${train_set} -> $datadir/${train_set}_sp"
            for factor in ${speed_perturb_factors}; do
                if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                    scripts/utils/perturb_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" \
                        "${factor}" "$datadir/${train_set}" "$datadir/${train_set}_sp${factor}"
                    _dirs+="$datadir/${train_set}_sp${factor} "
                else
                    # If speed factor is 1, same as the original
                    _dirs+="$datadir/${train_set} "
                fi
            done
            utils/combine_data.sh --extra_files "${utt_extra_files}" "$datadir/${train_set}_sp" ${_dirs}
            for extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py $datadir/"${train_set}_sp"/${extra_file} >$datadir/"${train_set}_sp"/${extra_file}.tmp
                mv $datadir/"${train_set}_sp"/${extra_file}.tmp $datadir/"${train_set}_sp"/${extra_file}
            done
        else
            log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: $datadir/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                #for dset in ${train_set}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print $datadir/"${dset}" "${data_feats}${_suf}/${dset}"

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls $datadir/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                echo "${expand_utt_extra_files}"
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}" "${data_feats}${_suf}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done

                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e $datadir/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments $datadir/${dset}/segments "
                fi
                if "${save_wav}"; then
                    _opts+="--save-wav true "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "$datadir/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

                echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank_pitch ]; then
            log "[Require Kaldi] Stage 3: ${feats_type} extract: $datadir/ -> ${data_feats}"

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # 1. Copy datadir
                utils/copy_data_dir.sh --validate_opts --non-print $datadir/"${dset}" "${data_feats}${_suf}/${dset}"

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls $datadir/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done

                # 2. Feature extract
                _nj=$(min "${nj}" "$(wc <"${data_feats}${_suf}/${dset}/utt2spk" -l)")
                steps/make_fbank_pitch.sh --nj "${_nj}" --cmd "${train_cmd}" "${data_feats}${_suf}/${dset}"
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}*" "${data_feats}${_suf}/${dset}"

                # 3. Derive the the frame length and feature dimension
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                # 4. Write feats_dim
                head -n 1 "${data_feats}${_suf}/${dset}/feats_shape" | awk '{ print $2 }' |
                    cut -d, -f2 >${data_feats}${_suf}/${dset}/feats_dim

                # 5. Write feats_type
                echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
            done

        elif [ "${feats_type}" = fbank ]; then
            log "Stage 3: ${feats_type} extract: $datadir/ -> ${data_feats}"
            log "${feats_type} is not supported yet."
            exit 1

        elif [ "${feats_type}" = extracted ]; then
            log "Stage 3: ${feats_type} extract: $datadir/ -> ${data_feats}"
            # Assuming you don't have wav.scp, but feats.scp is created by local/data.sh instead.

            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                # Generate dummy wav.scp to avoid error by copy_data_dir.sh
                awk <$datadir/"${dset}"/cmvn.scp ' { print($1,"<DUMMY>") }' >$datadir/"${dset}"/wav.scp
                utils/copy_data_dir.sh --validate_opts --non-print $datadir/"${dset}" "${data_feats}${_suf}/${dset}"

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                for extra_file in ${utt_extra_files}; do
                    # with regex to support multi-references
                    for single_file in $(ls $datadir/"${dset}"/${extra_file}*); do
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                utils/fix_data_dir.sh --utt_extra_files "${expand_utt_extra_files}*" "${data_feats}${_suf}/${dset}"
                for extra_file in ${expand_utt_extra_files}; do
                    LC_ALL=C sort -u -k1,1 "${data_feats}${_suf}/${dset}/${extra_file}" -o "${data_feats}${_suf}/${dset}/${extra_file}"
                done

                # Derive the the frame length and feature dimension
                _nj=$(min "${nj}" "$(wc <"${data_feats}${_suf}/${dset}/utt2spk" -l)")
                scripts/feats/feat_to_shape.sh --nj "${_nj}" --cmd "${train_cmd}" \
                    "${data_feats}${_suf}/${dset}/feats.scp" "${data_feats}${_suf}/${dset}/feats_shape"

                pyscripts/feats/feat-to-shape.py "scp:head -n 1 ${data_feats}${_suf}/${dset}/feats.scp |" - |
                    awk '{ print $2 }' | cut -d, -f2 >"${data_feats}${_suf}/${dset}/feats_dim"

                echo "${feats_type}" >"${data_feats}${_suf}/${dset}/feats_type"
            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi

    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short $datadir: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            for utt_extra_file in ${utt_extra_files}; do
                cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
            done
            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                awk <"${data_feats}/org/${dset}/utt2num_samples" -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
                utils/filter_scp.pl <"${data_feats}/org/${dset}/wav.scp" "${data_feats}/${dset}/utt2num_samples" \
                    >"${data_feats}/${dset}/wav.scp"
            else
                # Get frame shift in ms from conf/fbank.conf
                _frame_shift=
                if [ -f conf/fbank.conf ] && [ "$(grep <conf/fbank.conf -c frame-shift)" -gt 0 ]; then
                    # Assume using conf/fbank.conf for feature extraction
                    _frame_shift="$(grep <conf/fbank.conf frame-shift | sed -e 's/[-a-z =]*\([0-9]*\)/\1/g')"
                fi
                if [ -z "${_frame_shift}" ]; then
                    # If not existing, use the default number in Kaldi (=10ms).
                    # If you are using different number, you have to change the following value manually.
                    _frame_shift=10
                fi

                _min_length=$(python3 -c "print(int(${min_wav_duration} / ${_frame_shift} * 1000))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} / ${_frame_shift} * 1000))")

                cp "${data_feats}/org/${dset}/feats_dim" "${data_feats}/${dset}/feats_dim"
                awk <"${data_feats}/org/${dset}/feats_shape" -F, ' { print $1 } ' |
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length) print $0; }' \
                        >"${data_feats}/${dset}/feats_shape"
                utils/filter_scp.pl <"${data_feats}/org/${dset}/feats.scp" "${data_feats}/${dset}/feats_shape" \
                    >"${data_feats}/${dset}/feats.scp"
            fi

            # Remove empty text
            for utt_extra_file in ${utt_extra_files}; do
                awk <"${data_feats}/org/${dset}/${utt_extra_file}" ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/${utt_extra_file}"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"
            for utt_extra_file in ${utt_extra_files}; do
                python pyscripts/utils/remove_duplicate_keys.py ${data_feats}/${dset}/${utt_extra_file} \
                    >${data_feats}/${dset}/${utt_extra_file}.tmp
                mv ${data_feats}/${dset}/${utt_extra_file}.tmp ${data_feats}/${dset}/${utt_extra_file}
            done
        done

        # # shellcheck disable=SC2002
        # cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' \
        #     > "${data_feats}/lm_train.${src_lang}.${tgt_case}.${tgt_lang}.txt"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: Merge the wav.scp for the raw wav files to be decoded."
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            cat ${data_feats}${_suf}/${dset}/wav/*/wav.scp >${data_feats}${_suf}/${dset}/wav_raw.scp
        done
    fi
else
    log "Skip the stages for data preparation"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Run (distributed) ASR inference on the dev/test data."
    for dset in ${valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
    # for dset in ${test_sets}; do
        if [ "${dset}" = "${valid_set}" ]; then
            _suf="/org"
        else
            _suf=""
        fi
        _dsetdir=${data_feats}${_suf}/${dset}
        _dir="${st_exp}/${src_lang}/${dset}"
        _logdir="${st_exp}/logdir/inference_asr/${src_lang}/${dset}"
        mkdir -p "${_logdir}"

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${_dsetdir}/wav_raw.scp -l)")

        key_file=${_dsetdir}/wav_raw.scp
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Inference started... log: '${_logdir}/decode.*.log'"

        opts=
        if [ "${framework}" == "huggingface" ]; then
            _hf_dset="${hf_datadir}/${src_lang}.${dset}"
            opts+=" --dset ${_hf_dset} "
            inference_tool="pyscripts/utils/hf_whisper_inference.py"
        else
            inference_tool="pyscripts/utils/whisper_inference.py"
        fi

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        # shellcheck disable=SC2046,SC2086
        ${cuda_cmd} --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
            ${inference_tool} \
            --keyfile ${_logdir}/decode.JOB.scp \
            --src-lang ${src_lang} \
            --tgt-lang ${tgt_lang} \
            --output_dir ${_logdir}/output.JOB \
            --model_name ${model_name} ${opts}

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/text"
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Run evaluation on the ASR decoded data."

    # Note that we assume the evaluation code is available in the path
    for dset in ${valid_set} ${test_sets}; do
        # for dset in ${valid_set}; do
        # for dset in ${test_sets}; do
        log "Running evaluation on ${dset}"
        eval_script=run-asr-eval.sh

        _dir="${st_exp}/${src_lang}/${dset}"
        _asr_hyp="${PWD}/${_dir}/text"
        _dset=$(echo "${dset}" | sed 's/_test$//')

        opts=
        if [ "${src_lang}" == "ara" ]; then
            opts+=" --arabic true "
        fi

        cd evaluation
        ${eval_script} \
            --src_lang ${src_lang} \
            --hyp_asr "${_asr_hyp}" \
            --sclite ${sclite_path} \
            --model_tag ${model_name} \
            --dset "${_dset}" \
            --framework "${framework}" ${opts}
        cd -
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Run (distributed) ST inference on the dev/test data."
    # for dset in ${valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
        for dset in ${test_sets}; do
        if [ "${dset}" = "${valid_set}" ]; then
            _suf="/org"
        else
            _suf=""
        fi
        _dsetdir=${data_feats}${_suf}/${dset}
        _dir="${st_exp}/${src_lang}/${dset}"
        _logdir="${st_exp}/logdir/inference_st/${src_lang}/${dset}"
        mkdir -p "${_logdir}"

        # 1. Split the key file
        _nj=$(min "${inference_nj}" "$(wc <${_dsetdir}/wav_raw.scp -l)")

        key_file=${_dsetdir}/wav_raw.scp
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/decode.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "Inference started... log: '${_logdir}/decode.*.log'"

        opts=
        if [ "${framework}" == "huggingface" ]; then
            _hf_dset="${hf_datadir}/${src_lang}.${dset}"
            opts+=" --dset ${_hf_dset} "
            inference_tool="pyscripts/utils/hf_whisper_inference.py"
        else
            inference_tool="pyscripts/utils/whisper_inference.py"
        fi

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.
        # shellcheck disable=SC2046,SC2086
        ${cuda_cmd} --mem 16G --gpu 1 JOB=1:"${_nj}" "${_logdir}"/decode.JOB.log \
            ${inference_tool} \
            --keyfile ${_logdir}/decode.JOB.scp \
            --src-lang ${src_lang} \
            --tgt-lang ${tgt_lang} \
            --output_dir ${_logdir}/output.JOB \
            --model_name ${model_name} \
            --task "translate" ${opts}

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/text"
        done | LC_ALL=C sort -k1 >"${_dir}/st.text"
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    log "Stage 9: Run evaluation on the ST decoded data."

    # Note that we assume the evaluation code is available in the path
    # for dset in ${valid_set} ${test_sets}; do
    # for dset in ${valid_set}; do
        for dset in ${test_sets}; do
        log "Running evaluation on ${dset}"

        if [ "${dset}" = "${valid_set}" ]; then
            eval_script=run-devset-eval.sh
        elif [ "${dset}" = "fleurs_test" ]; then
            eval_script=run-ood-eval.sh
        else
            eval_script=run-testset-eval.sh
        fi

        opts=
        if [ "${src_lang}" == "ara" ]; then
            opts+=" --arabic true "
        fi

        _dir="${st_exp}/${src_lang}/${dset}"
        _dset=$(echo "${dset}" | sed 's/_test$//')
        st_hyp_file="${PWD}/${_dir}/st.text"
        cd evaluation
        $eval_script \
            --src_lang ${src_lang} \
            --hyp_mt ${st_hyp_file} \
            --model_tag ${model_name} \
            --score_dir scores \
            --dset "${_dset}" \
            --framework "${framework}" ${opts}
        cd -
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
