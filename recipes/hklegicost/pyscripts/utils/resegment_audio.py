#!/usr/bin/env python
from tqdm import tqdm
import local.data_prep.stm as stm
import argparse
from pathlib import Path
import librosa
import soundfile as sf


def reseg(input_file: Path, output_file: Path, base_dir: Path, audio_dir: Path):
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    utts = [stm.parse_StmUtterance(line) for line in lines]
    with open(output_file, 'w') as f:
        for utt in tqdm(utts):
            filename = utt.filename
            uttid = utt.utterance_id(stereo=True)
            audio_path = base_dir / filename
            new_audio_path = audio_dir / f"{uttid}.wav"
            
            # Re-segment the audio based on the supervisions
            start_time = utt.start_time
            stop_time = utt.stop_time
            wav, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=stop_time-start_time)
            # Do not save the wav file if it exsits already
            if not new_audio_path.exists():
                sf.write(new_audio_path, wav, sr)

            # Generate the new stm utterance
            new_utt = stm.StmUtterance(
                filename=new_audio_path,
                channel=utt.channel,
                speaker=utt.speaker,
                start_time=0,
                stop_time=stop_time-start_time,
                transcript=utt.transcript
            )
            
            print(new_utt, file=f)


def main():
    parser = argparse.ArgumentParser(description='Convert stm to scp')
    parser.add_argument('-i', '--input-file', type=Path,
                        required=True, help='Input stm file directory.')
    parser.add_argument('-o', '--output-file', type=Path,
                        required=True, help='Output scp file directory.')
    parser.add_argument('--audio-dir', type=Path,
                        required=True, help='Target directory for storing the segmented audio.')
    parser.add_argument('-b', '--base-dir', type=Path,
                        default="", help='Base directory for locating the audio.')
    args = parser.parse_args()

    reseg(input_file=args.input_file, audio_dir=args.audio_dir,
          output_file=args.output_file, base_dir=args.base_dir)


if __name__ == '__main__':
    main()
