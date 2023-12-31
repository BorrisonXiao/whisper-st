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
score_dir=
src_lang=
ref_mt=
hyp_mt=
ref_asr=
hyp_asr=
arabic=false
python=python3
sclite=sclite
comet=none
comet_model=/exp/mmartindale/scale23/shared/comet_models/comet/checkpoints/model.ckpt
cer=false

help_message=$(cat << EOF
Usage: $0 --score_dir <path_to_dir> --ref_mt <path_to_ref_file> --hyp_mt <path_to_hyp_file>

Options:
    --score_dir     # Directory to store results.
    --src_lang      # Source language trigraph.
    --ref_mt        # Reference file for translation. STM format.
    --hyp_mt        # Hypothesis file for translation. STM format.
    --ref_asr       # Reference file for ASR. STM format.
    --hyp_asr       # Hypothesis file for ASR. STM format.
    --arabic        # Choose Arabic normalization for ASR (default="${arabic}").
    --python        # Specify python command (default="${python}").
    --sclite        # Sclite binary (default="${sclite}").
    --comet			# Type of COMET score to report [none(default)|segment|system]
    --comet_model	# Comet model checkpoint to use for scoring
EOF
)

log "$0 $*"

scriptdir="$( dirname -- "$BASH_SOURCE"; )";
pyscripts=$scriptdir/pyscripts
utils=$scriptdir/utils

run_args=$($pyscripts/utils/print_args.py $0 "$@")
. $utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

if [ ! -d ${score_dir} ]; then
    mkdir -p ${score_dir}
fi

run_asr=false
run_mt=false
# Check for MT STM files, both must exist, set run_mt=true
if [ "$ref_mt" != "" ] && [ "$hyp_mt" != "" ]; then
    run_mt=true
fi

if ([ "$ref_mt" == "" ] && [ "$hyp_mt" != "" ]) || ([ "$ref_mt" != "" ] && [ "$hyp_mt" == "" ]); then
    echo "WARNING: --ref_mt and --hyp_mt must both be set for translation eval to run"
fi

# Check comet settings
if [ "$comet" != "none" ]; then
    if [[ "$comet" == "seg"* ]]; then
        comet_out=${score_dir}/comet.seg.stm
    elif [[ "$comet" == "sys"* ]]; then
        comet_out=${score_dir}/comet.sys.txt
    else
        echo "WARNING: $comet is not a valid comet level. Assuming system level."
        comet_out=${score_dir}/comet.sys.txt
    fi
    if [ $run_mt == false ]; then
        echo "WARNING: --ref_mt and --hyp_mt must both be set for comet eval to run"
        comet="none"
    fi
    if [ "$ref_asr" == "" ]; then
        echo "WARNING: --ref_asr must be set for comet eval to run"
        comet="none"
    fi
fi

# Check for ASR STM files, both must exist, set run_asr=true
if [ "$ref_asr" != "" ] && [ "$hyp_asr" != "" ]; then
    run_asr=true
fi

if ([ "$ref_asr" == "" ] && [ "$hyp_asr" != "" ]) || ([ "$ref_asr" != "" ] && [ "$hyp_asr" == "" ]); then
    echo "WARNING: --ref_asr and --hyp_asr must both be set for ASR eval to run"
fi

arabic_norm=""
if "$arabic"; then
    arabic_norm="--arabic"
fi

# clean up old files
if [ -f ${score_dir}/result.lc.rm.txt ]; then
    rm ${score_dir}/result.lc.rm.txt
fi

if [ -f ${score_dir}/result.lc.txt ]; then
    rm ${score_dir}/result.lc.txt
fi

if [ "$comet" != "none" ] && [ -f $comet_out ]; then
    rm $comet_out
fi


# Apply GLM to STM files here, for each input STM if it exists
for input_file in $ref_mt $hyp_mt $ref_asr $hyp_asr; do

    case $input_file in
        $ref_mt)
            file_copy=ref_mt.stm
            glm_lang=eng;;
        $hyp_mt)
            file_copy=hyp_mt.stm
            glm_lang=eng;;
        $ref_asr)
            file_copy=ref_asr.stm
            glm_lang=${src_lang};;
        $hyp_asr)
            file_copy=hyp_asr.stm
            glm_lang=${src_lang};;
    esac

    echo "Copying ..." #$input_file into $score_dir/$file_copy"
    cp $input_file $score_dir/$file_copy

    echo "Normalize STM times ... " #$score_dir/$file_copy"
    $python $pyscripts/utils/normalize_stm_times.py $score_dir/$file_copy $score_dir/${file_copy}.norm 2> ${score_dir}/normalize.err.log

    echo "Applying GLM rules ..." #to $score_dir/$file_copy"
    $python $pyscripts/utils/apply_glm_rules.py $score_dir/$file_copy.norm $utils/glm.${glm_lang} stm $score_dir/${file_copy}.glm 2> ${score_dir}/glm.err.log
done

# Check if ${run_asr}. Run ASR eval here.
if [ ${run_asr} == true ]; then
    echo "Running ASR score"
    echo "ASR results" > ${score_dir}/result.lc.rm.txt
    _opts=
    if "${cer}"; then
        _opts+=" --cer "
    fi
    $python $pyscripts/utils/stm_wer.py $sclite $score_dir/ref_asr.stm.glm $score_dir/hyp_asr.stm.glm $score_dir $arabic_norm ${_opts} >> ${score_dir}/result.lc.rm.txt
    echo "" >> ${score_dir}/result.lc.rm.txt
fi

# Check if ${run_mt}. Run MT eval here.
if [ ${run_mt} == true ]; then
    echo "Running translation score"

    echo "Check if hyp utterances missing and re-add as empty lines"
    $python $pyscripts/utils/align_stms.py ${score_dir}/hyp_mt.stm.glm ${score_dir}/ref_mt.stm.glm ${score_dir}/hyp_mt.stm.glm.aligned

    # Sort STM on both hyp and ref to make sure bitext is aligned and convert to mt format
    SRC=
    if [ "$comet" != "none" ]; then
       SRC=ref_asr.stm.glm
    fi
    for input_file in ref_mt.stm.glm hyp_mt.stm.glm.aligned $SRC; do
        case $input_file in
            "ref_mt.stm.glm")
                sorted_file=ref.tc.sorted
                bitext_file=ref.tc;;
            "hyp_mt.stm.glm.aligned")
                sorted_file=hyp.tc.sorted
                bitext_file=hyp.tc;;
            "ref_asr.stm.glm")
                sorted_file=src.tc.sorted
                bitext_file=src.tc;;
        esac

        echo "Sorting input file ... " #$input_file"
        sort -t ' ' -k1,5 ${score_dir}/${input_file} > ${score_dir}/${sorted_file}

        echo "Creating bitext ... " #${score_dir}/${bitext_file} ${score_dir}/${sorted_file}"
        cut -d' ' -f7- "${score_dir}/${sorted_file}" > "${score_dir}/${bitext_file}"

    done
    # Clean mt file (remove punctuations, remove non-speech tokens). Lowercase is done via the "-lc" flag on sacrebleu
    SRC=
    if [ "$comet" != "none" ]; then
        SRC=src
    fi
    for input_type in ref hyp $SRC; do
        echo "Removing punctuatons ... " #for ${score_dir}/${input_type}.tc.rm"
        $utils/remove_punctuation.pl < "${score_dir}/${input_type}.tc" > "${score_dir}/${input_type}.tc.rm"
    done

    # Run COMET evaluation
    if [ "$comet" != "none" ]; then
        if [[ "$comet" == "seg"* ]]; then
            comet=""
            echo "Segment-level COMET scores with model $comet_model" >> $comet_out
        elif [[ "$comet" == "sys"* ]]; then
            echo "System-level COMET score with model $comet_model" >> $comet_out
            comet="--only_system"
        else
            echo "WARNING: '$comet' is not a valid COMET type. Assuming system-only."
            echo "System-level COMET score with model $comet_model" >> $comet_out
            comet="--only_system"
        fi        
        #echo "${python} $pyscripts/comet_evaluation_scale23.py -m $comet_model -s ${score_dir}/src.tc.rm -r ${score_dir}/ref.tc.rm -t ${score_dir}/hyp.tc.rm $comet >> $comet_out"
        ${python} $pyscripts/comet_evaluation_scale23.py \
                  -m $comet_model \
                  -s ${score_dir}/src.tc.rm \
                  -r ${score_dir}/ref.tc.rm \
                  -t ${score_dir}/hyp.tc.rm \
                  -i ${score_dir}/hyp.tc.sorted \
                  $comet >> $comet_out
    fi

    # Run sacrebleu
    echo "Case insensitive BLEU result (single-reference)" >> ${score_dir}/result.lc.rm.txt
    sacrebleu -lc "${score_dir}/ref.tc.rm" \
            -i "${score_dir}/hyp.tc.rm" \
            -m bleu chrf ter \
            -b -w 4 \
            >> ${score_dir}/result.lc.rm.txt

    echo "" >> ${score_dir}/result.lc.rm.txt

    #echo "Score summary"
    #${python} $pyscripts/utils/process_result_file.py --score-only ${score_dir}/result.lc.rm.txt
fi

cat ${score_dir}/result.lc.rm.txt
if [ "$comet" != "none" ]; then
	tail -n 1 $comet_out
fi
