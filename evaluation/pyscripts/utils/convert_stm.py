import argparse

from collections import defaultdict
import os

def _get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("stm")
    parser.add_argument("score_dir")
    parser.add_argument("text_file")

    return parser

def parse_stm(stm):
    #/export/common/data/corpora/LDC/LDC2010S01/DISC1/data/speech/20051028_180633_356_fsp.sph 1 100741 13.7647058824 14.9243697479 <O> In Philadelphia.

    text_list = []
    utt2spk_dict = defaultdict(list)

    with open(stm, 'r', encoding='utf-8') as f:
        for line in f:
            splitted_line = line.strip().split()

            audio_path = splitted_line[0]
            channel = splitted_line[1]
            spk = splitted_line[2]
            start = splitted_line[3]
            end = splitted_line[4]
            text = splitted_line[6:]

            if channel in ['a', 'A', '0']:
                channel = 'A'
            else:
                channel = 'B'

            basename = os.path.splitext(os.path.basename(audio_path))[0]
            start_str = float(start) * 100
            end_str = float(end) * 100

            utt_id = "%s-%s-%s_%08.0f_%08.0f" % (spk, basename, channel, start_str, end_str)

            text_list.append("%s %s" % (utt_id, " ".join(text)))
            utt2spk_dict[utt_id].append(spk)

    # Sort speakers
    for utt_id in utt2spk_dict:
        utt2spk_dict[utt_id].sort()

    return text_list, utt2spk_dict

def main(args):
    stm = args.stm
    score_dir = args.score_dir
    text_file = args.text_file

    text_list, utt2spk_dir = parse_stm(stm)

    with open("%s/%s" % (score_dir, text_file), 'w', encoding='utf-8') as f, open("%s/utt2spk" % score_dir, 'w', encoding='utf-8') as f2:
        for text_line in text_list:
            f.write("%s\n" % text_line)

        for utt_id in utt2spk_dir:
            f2.write("%s " % utt_id)
            f2.write("%s\n" % " ".join(utt2spk_dir[utt_id]))


if __name__ == "__main__":
    args = _get_parser().parse_args()

    main(args)