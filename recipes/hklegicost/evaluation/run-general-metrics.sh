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

# General options
score_dir=      # Top directory to store results
run_st=false    # Run ST scoring
run_asr=false   # Run ASR scoring
run_mt=false    # Run MT scoring
run_single_ref=true # Run single reference BLEU
run_multi_ref=false # Run multi reference BLEU
python=python3

# Options for ST
st_ref_file=    # Reference file for ST
st_hyp_file=    # Hypothesis file for ST
st_utt2spk=     # utt2spk for ST

# Options for ASR
asr_ref_file=   # Reference file for ASR
asr_hyp_file=   # Hypothesis file for ASR
asr_utt2spk=    # utt2spk for ASR
use_cer=false   # If true use WER, otherwise use CER

# Options for MT
mt_ref_file=    # Reference file for MT
mt_hyp_file=    # Hypothesis file for MT

# Note: st and mt files are essentially the same, the main difference is that st files assume utt2spk to exist

help_message=$(cat << EOF
Usage: $0
EOF
)

log "$0 $*"

run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

if "${run_st}"; then
    st_score_dir=${score_dir}/score_bleu_st
    mkdir -p ${st_score_dir}

    # Run cleaner
    # Clean hypothesis file
    ./scripts/clean_data.sh --input_file $st_hyp_file \
                    --utt2spk $st_utt2spk \
                    --input_type "hyp" \
                    --score_dir $st_score_dir

    # Run single reference scoring
    if "${run_single_ref}"; then
        echo "Running Single Reference"
        # Run cleaner
        # Clean single-reference file
        ./scripts/clean_data.sh --input_file $st_ref_file \
                        --utt2spk $st_utt2spk \
                        --score_dir $st_score_dir \
                        --use_cleaner true

        # Run scoring
        echo "Case insensitive BLEU result (single-reference)" > ${st_score_dir}/result.lc.txt
        sacrebleu -lc "${st_score_dir}/ref.trn.detok.lc.rm" \
                  -i "${st_score_dir}/hyp.trn.detok.lc.rm" \
                  -m bleu chrf ter \
                  >> ${st_score_dir}/result.lc.txt
    fi

    if "${run_multi_ref}"; then
        echo "Running Multi Reference"
        # Run cleaner
        # Clean multi-reference file
        multi_references=$(ls ${st_ref_file}.* || echo "")
        if [ "${multi_references}" != "" ]; then
            multi_refs=""
            for multi_reference in ${multi_references}; do
                ref_idx="${multi_reference##*.}"
                ./scripts/clean_data.sh --input_file ${st_ref_file}.${ref_idx} \
                    --utt2spk $st_utt2spk \
                    --score_dir $st_score_dir \
                    --use_cleaner true \
                    --file_idx $ref_idx

                # Create list of multi references
                multi_refs="${multi_refs} ${st_score_dir}/ref.trn.detok.lc.rm.${ref_idx}"
            done

            # Run scoring
            echo "Case insensitive BLEU result (multi-references)" > ${st_score_dir}/result.lc.txt
            sacrebleu -lc $multi_refs \
                    -i "${st_score_dir}/hyp.trn.detok.lc.rm" \
                    -m bleu chrf ter \
                    >> ${st_score_dir}/result.lc.txt
        fi
    fi
fi

if "${run_asr}"; then
    if ! "${use_cer}"; then
        asr_score_dir=${score_dir}/score_wer
        mkdir -p ${asr_score_dir}
    else
        asr_score_dir=${score_dir}/score_cer
        mkdir -p ${asr_score_dir}
    fi

    # Run cleaner
    # Clean reference file
    ./scripts/clean_data.sh --input_file $asr_ref_file \
                    --utt2spk $asr_utt2spk \
                    --score_dir $asr_score_dir \
                    --use_cleaner true

    # Clean hypothesis file
    ./scripts/clean_data.sh --input_file $asr_hyp_file \
                    --utt2spk $asr_utt2spk \
                    --input_type "hyp" \
                    --score_dir $asr_score_dir

    # Run scoring
    sclite \
        -r "${asr_score_dir}/ref.trn.org" trn \
        -h "${asr_score_dir}/hyp.trn.org" trn \
        -i rm -o all stdout > "${asr_score_dir}/result.lc.txt"
fi

if "${run_mt}"; then
    mt_score_dir=${score_dir}/score_bleu_mt
    mkdir -p ${mt_score_dir}

    # Run cleaner
    # Note: MT scores might only need to be detokenized.
    # Clean hypothesis file
    detokenizer.perl -l en -q < "${mt_hyp_file}" > "${mt_score_dir}/hyp.trn.detok"

    remove_punctuation.pl < "${mt_score_dir}/hyp.trn.detok" > "${mt_score_dir}/hyp.trn.detok.lc.rm"

    if "${run_single_ref}"; then
        # Run cleaner
        # Clean single-reference file
        detokenizer.perl -l en -q < "${mt_ref_file}" > "${mt_score_dir}/ref.trn.detok"

        remove_punctuation.pl < "${mt_score_dir}/ref.trn.detok" > "${mt_score_dir}/ref.trn.detok.lc.rm"

        # Run scoring
        echo "Case insensitive BLEU result (single-reference)" > ${mt_score_dir}/result.lc.txt
        sacrebleu -lc "${mt_score_dir}/ref.trn.detok.lc.rm" \
                  -i "${mt_score_dir}/hyp.trn.detok.lc.rm" \
                  -m bleu chrf ter \
                  >> ${mt_score_dir}/result.lc.txt
    fi

    if "${run_multi_ref}"; then
        # Run cleaner
        # Clean multi-reference file
        multi_references=$(ls ${mt_ref_file}.* || echo "")
        if [ "${multi_references}" != "" ]; then
            multi_refs=""
            for multi_reference in ${multi_references}; do
                ref_idx="${multi_reference##*.}"

                detokenizer.perl -l en -q < "${mt_ref_file}.${ref_idx}" > "${mt_score_dir}/ref.trn.detok.${ref_idx}"

                remove_punctuation.pl < "${mt_score_dir}/ref.trn.detok.${ref_idx}" > "${mt_score_dir}/ref.trn.detok.lc.rm.${ref_idx}"

                # Create list of multi references
                multi_refs="${multi_refs} ${mt_score_dir}/ref.trn.detok.lc.rm.${ref_idx}"
            done

            # Run scoring
            echo "Case insensitive BLEU result (multi-references)" > ${mt_score_dir}/result.lc.txt
            sacrebleu -lc $multi_refs \
                      -i "${mt_score_dir}/hyp.trn.detok.lc.rm" \
                      -m bleu chrF ter \
                      >> ${mt_score_dir}/result.lc.txt
        fi
    fi
fi

echo ""

if "${run_st}"; then
    echo "ST scores"
    ${python} pyscripts/utils/process_result_file.py ${st_score_dir}/result.lc.txt
fi
if "${run_asr}"; then
    echo "ASR scores"
    ${python} pyscripts/utils/process_result_file.py --wer ${asr_score_dir}/result.lc.txt
fi
if "${run_mt}"; then
    echo "MT scores"
    ${python} pyscripts/utils/process_result_file.py ${mt_score_dir}/result.lc.txt
fi