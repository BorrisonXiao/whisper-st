#!/usr/bin/env python
from tqdm import tqdm
import local.shared.stm as stm
import argparse
from pathlib import Path
import librosa
import soundfile as sf
from resegment_audio import reseg


def resegs


def main():
    parser = argparse.ArgumentParser(description='Convert stm to scp')
    parser.add_argument('-i', '--input-dir', type=Path,
                        required=True, help='Input stm file directory.')
    parser.add_argument('-o', '--output-dir', type=Path,
                        required=True, help='Output scp file directory.')
    parser.add_argument('--audio-dir', type=Path,
                        required=True, help='Target directory for storing the segmented audio.')
    args = parser.parse_args()

    reseg(input_file=args.input_file, audio_dir=args.audio_dir,
          output_file=args.output_file, base_dir=args.base_dir)


if __name__ == '__main__':
    main()
