#!/usr/bin/env python3

from transformers import pipeline
from transformers import MarianTokenizer
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import re

long_to_short = {
  "ara": "ar",
  "cmn": "zh",
  "kor": "ko",
  "rus": "ru",
  "spa": "es",
}

lang2iter = {
    "ara": 63000,
    "cmn": 43500,
    "kor": 2500,
    "rus": 3500, 
    "spa": 48000,
}

def _parse_stm_line(l):
    '''
        Parse a line of the stm file. The format is 

        audiopath channel speaker start end label text

        In the event that there is not text (it is empty for the segment), then
        we catch that case. 
    '''
    try:
        path, channel, speaker, start, end, lbl, text = l.strip().split(None, 6)
    except ValueError:
        path, channel, speaker, start, end, lbl = l.strip().split(None, 5)
        text = ''
    recoid = path.replace("//", "/")
    text = re.sub(r"\-\-+", "\-", text)
    return {
        'recoid': recoid,
       'channel': channel,
       'speaker': speaker,
       'start': float(start),
       'end': float(end),
       'text': text,
       'label': lbl,
    }

def setup_argparse():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--source", type=str, choices=long_to_short.keys(), required=True)
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-r", "--reference", type=str)
    parser.add_argument("-o", "--output", type=FileType("w"), default="-")
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    return parser


if __name__ == "__main__":
    args = setup_argparse().parse_args()
    hyp_stm, ref_stm = {}, {}
    with open(args.input, 'r', encoding='utf-8') as f:
        for l in f:
            stm = _parse_stm_line(l)
            recoid, channel = stm['recoid'], stm['channel']
            speaker, start, end = stm['speaker'], stm['start'], stm['end']
            hyp_stm[(recoid, channel, speaker, start, end)] = stm['text'] 

    with open(args.reference, 'r', encoding='utf-8') as f:
        for l in f:
            stm = _parse_stm_line(l)
            recoid, channel = stm['recoid'], stm['channel']
            speaker, start, end = stm['speaker'], stm['start'], stm['end']
            ref_stm[(recoid, channel, speaker, start, end)] = stm['text'] 

    src = long_to_short[args.source]
    #mt = pipeline(f"translation_{src}_to_en", f"Helsinki-NLP/opus-mt-{src}-en", device=args.device)
    tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-en")
    mt = pipeline(f"translation_{src}_to_en", model=f"/exp/erippeth/scale_baselines/opus_finetuned/{args.source}/checkpoint-{lang2iter[args.source]}", 
        tokenizer=tokenizer, device=args.device)
    # truncate long examples to model's max_length (200 subword tokens)
    mt.tokenizer.truncation = True
    inputs = [v for k, v in sorted(hyp_stm.items(), key=lambda x: x[0])]
    input_keys = sorted(hyp_stm.keys())
    outputs = mt(inputs, num_beams=5, batch_size=args.batch_size)

    with args.output as fout:
        for k, b in zip(input_keys, outputs):
            text = b['translation_text']
            stm_line = f'{k[0]} {k[1]} {k[2]} {k[3]} {k[4]} <O> {text}'
            print(stm_line, file=fout)
