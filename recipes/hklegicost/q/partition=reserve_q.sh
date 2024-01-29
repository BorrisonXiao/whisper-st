#!/bin/bash
cd /home/cxiao7/research/whisper-st/recipes/hklegicost
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
--mem 16G --gpu 1 JOB=1:4 exp/large-v2/logs/inference_asr/dev-asr-0//decode.JOB.log pyscripts/utils/hf_whisper_inference.py --keyfile exp/large-v2/logs/inference_asr/dev-asr-0//decode.JOB.scp --src-lang cmn --tgt-lang cmn --output_dir exp/large-v2/logs/inference_asr/dev-asr-0//output.JOB --batch-size 2 --model_name large-v2 --dset dump/raw/dev-asr-0 
EOF
) >partition=reserve_q
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>partition=reserve_q
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( --mem 16G --gpu 1 JOB=1:4 exp/large-v2/logs/inference_asr/dev-asr-0//decode.JOB.log pyscripts/utils/hf_whisper_inference.py --keyfile exp/large-v2/logs/inference_asr/dev-asr-0//decode.JOB.scp --src-lang cmn --tgt-lang cmn --output_dir exp/large-v2/logs/inference_asr/dev-asr-0//output.JOB --batch-size 2 --model_name large-v2 --dset dump/raw/dev-asr-0  ) &>>partition=reserve_q
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>partition=reserve_q
echo '#' Accounting: end_time=$time2 >>partition=reserve_q
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>partition=reserve_q
echo '#' Finished at `date` with status $ret >>partition=reserve_q
[ $ret -eq 137 ] && exit 100;
touch ./q/done.2277661
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --ntasks-per-node=1  -l account=reserve -p shared  --open-mode=append -e ./q/partition=reserve_q -o ./q/partition=reserve_q  /home/cxiao7/research/whisper-st/recipes/hklegicost/./q/partition=reserve_q.sh >>./q/partition=reserve_q 2>&1
