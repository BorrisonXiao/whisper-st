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
dset=
split=
ref_dir=
hyp_mt=
python=python3
comet=none
comet_model=/exp/mmartindale/scale23/shared/comet_models/comet/checkpoints/model.ckpt
no_glm=false

help_message=$(
    cat <<EOF
Usage: $0 --score_dir <path_to_dir> --ref_mt <path_to_ref_file> --hyp_mt <path_to_hyp_file>

Options:
    --score_dir     # Directory to store results.
    --src_lang      # Source language trigraph.
    --ref_dir       # Reference file for translation. STM format.
    --hyp_mt        # Hypothesis file for translation. STM format.
    --ref_asr       # Reference file for ASR. STM format.
    --hyp_asr       # Hypothesis file for ASR. STM format.
    --python        # Specify python command (default="${python}").
    --comet			# Type of COMET score to report [none(default)|segment|system]
    --comet_model	# Comet model checkpoint to use for scoring
EOF
)

log "$0 $*"

scriptdir="$(dirname -- "$BASH_SOURCE")"
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
fi

res_file=${score_dir}/result.multi.lc.rm.txt
if "${no_glm}"; then
    res_file=${score_dir}/result.multi.noglm.lc.rm.txt
fi

# clean up old files
if [ -f ${res_file} ]; then
    rm ${res_file}
fi

if [ "$comet" != "none" ] && [ -f $comet_out ]; then
    rm $comet_out
fi

# Apply GLM to all STM reference files in ref_dir
for input_file in "$ref_dir"/*; do
    # Skip files that are not ${dset}_${split}.en.*.stm
    if [[ $(basename "$input_file") != ${dset}_${split}.en.*.stm ]]; then
        continue
    fi

    # Get the digit part of the file name
    idx=$(basename "$input_file" | sed -r 's/.*\.en\.([0-9]+)\.stm/\1/')
    file_copy=ref_mt.${idx}.stm
    echo "Processing $file_copy"
    glm_lang=eng
    if "$no_glm"; then
        echo "No GLM rules applied"
        glm_lang=dummy
    fi

    echo "Copying ..." #$input_file into $score_dir/$file_copy"
    cp "$input_file" "$score_dir/$file_copy"

    echo "Normalize STM times ... " #$score_dir/$file_copy"
    $python $pyscripts/utils/normalize_stm_times.py "$score_dir/$file_copy" "$score_dir/${file_copy}.norm" 2>"${score_dir}/normalize.err.log"

    echo "Applying GLM rules ..." #to $score_dir/$file_copy"
    $python $pyscripts/utils/apply_glm_rules.py "$score_dir/$file_copy.norm" "$utils/glm.${glm_lang}" stm "$score_dir/${file_copy}.glm" 2>"${score_dir}/glm.err.log"

done

# Apply GLM to STM files here, for each input STM if it exists
for input_file in $hyp_mt; do

    file_copy=hyp_mt.stm
    glm_lang=eng
    if "$no_glm"; then
        echo "No GLM rules applied"
        glm_lang=dummy
    fi

    echo "Copying ..." #$input_file into $score_dir/$file_copy"
    cp $input_file $score_dir/$file_copy

    echo "Normalize STM times ... " #$score_dir/$file_copy"
    $python $pyscripts/utils/normalize_stm_times.py $score_dir/$file_copy $score_dir/${file_copy}.norm 2>${score_dir}/normalize.err.log

    echo "Applying GLM rules ..." #to $score_dir/$file_copy"
    $python $pyscripts/utils/apply_glm_rules.py $score_dir/$file_copy.norm $utils/glm.${glm_lang} stm $score_dir/${file_copy}.glm 2>${score_dir}/glm.err.log
done

# Run MT eval here.
echo "Running translation score"

# echo "Check if hyp utterances missing and re-add as empty lines"
# $python $pyscripts/utils/align_stms.py ${score_dir}/hyp_mt.stm.glm ${score_dir}/ref_mt.0.stm.glm ${score_dir}/hyp_mt.stm.glm.aligned
cp ${score_dir}/hyp_mt.stm.glm ${score_dir}/hyp_mt.stm.glm.aligned

# Sort STM on both hyp and ref to make sure bitext is aligned and convert to mt format
SRC=
if [ "$comet" != "none" ]; then
    SRC=ref_asr.stm.glm
fi
for input_file in hyp_mt.stm.glm.aligned $SRC; do
    case $input_file in
    "hyp_mt.stm.glm.aligned")
        sorted_file=hyp.tc.sorted
        bitext_file=hyp.tc
        ;;
    "ref_asr.stm.glm")
        sorted_file=src.tc.sorted
        bitext_file=src.tc
        ;;
    esac

    echo "Sorting input file ... " #$input_file"
    sort -t ' ' -k1,5 ${score_dir}/${input_file} >${score_dir}/${sorted_file}

    echo "Creating bitext ... " #${score_dir}/${bitext_file} ${score_dir}/${sorted_file}"
    cut -d' ' -f7- "${score_dir}/${sorted_file}" >"${score_dir}/${bitext_file}"
done

# for input_file in ref_mt_*.stm.glm; do
for input_file in $score_dir/ref_mt.*.stm.glm; do
    idx=$(basename "$input_file" | sed -r 's/.*\.([0-9]+)\.stm.glm/\1/')
    sorted_file=ref.tc.${idx}.sorted
    bitext_file=ref.tc.${idx}

    echo "Sorting input file ... " #$input_file"
    sort -t ' ' -k1,5 ${input_file} >${score_dir}/${sorted_file}

    echo "Creating bitext ... " #${score_dir}/${bitext_file} ${score_dir}/${sorted_file}"
    cut -d' ' -f7- "${score_dir}/${sorted_file}" >"${score_dir}/${bitext_file}"
done

# Clean mt file (remove punctuations, remove non-speech tokens). Lowercase is done via the "-lc" flag on sacrebleu
SRC=
if [ "$comet" != "none" ]; then
    SRC=src
fi
for input_type in hyp $SRC; do
    echo "Removing punctuatons ... " #for ${score_dir}/${input_type}.tc.rm"
    $utils/remove_punctuation.pl <"${score_dir}/${input_type}.tc" >"${score_dir}/${input_type}.tc.rm"
done
for input_file in $score_dir/ref.tc.*; do
    if [[ "$input_file" =~ \.([0-9]+)$ ]]; then
        idx=$(basename "$input_file" | sed -r 's/.*\.([0-9]+)$/\1/')
        echo "Removing punctuatons ... " #for ${input_file}.rm"
        $utils/remove_punctuation.pl <"${input_file}" >"${input_file}.rm"
    fi
done

# Run COMET evaluation
if [ "$comet" != "none" ]; then
    if [[ "$comet" == "seg"* ]]; then
        comet=""
        echo "Segment-level COMET scores with model $comet_model" >>$comet_out
    elif [[ "$comet" == "sys"* ]]; then
        echo "System-level COMET score with model $comet_model" >>$comet_out
        comet="--only_system"
    else
        echo "WARNING: '$comet' is not a valid COMET type. Assuming system-only."
        echo "System-level COMET score with model $comet_model" >>$comet_out
        comet="--only_system"
    fi
    #echo "${python} $pyscripts/comet_evaluation_scale23.py -m $comet_model -s ${score_dir}/src.tc.rm -r ${score_dir}/ref.tc.rm -t ${score_dir}/hyp.tc.rm $comet >> $comet_out"
    ${python} $pyscripts/comet_evaluation_scale23.py \
        -m $comet_model \
        -s ${score_dir}/src.tc.rm \
        -r ${score_dir}/ref.tc.rm \
        -t ${score_dir}/hyp.tc.rm \
        -i ${score_dir}/hyp.tc.sorted \
        $comet >>$comet_out
fi

# Run multi-bleu
echo "Writing results to ${res_file}"
echo "Case insensitive multi-BLEU result (single-reference)" >>${res_file}
./scripts/multi-bleu.pl -lc ${score_dir}/ref.tc.*.rm <${score_dir}/hyp.tc.rm >>${res_file}

cat ${res_file}
if [ "$comet" != "none" ]; then
    tail -n 1 $comet_out
fi
