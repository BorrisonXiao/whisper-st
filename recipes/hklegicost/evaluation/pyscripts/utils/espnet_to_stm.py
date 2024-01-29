import argparse

from collections import defaultdict
import os

def _get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("text")
    parser.add_argument("wav_scp")
    parser.add_argument("utt2spk")
    parser.add_argument("segments")
    parser.add_argument("reco2file_and_channel")
    parser.add_argument("output_stm")

    return parser


def create_stm(text, wav_scp, utt2spk, segments, reco2file_and_channel):
    stm_list = []

    text_dict = {}
    wav_dict = {}
    utt2spk_dict = {}
    segments_dict = defaultdict(list)
    reco2file_and_channel_dict = {}

    with open(text, 'r', encoding='utf-8') as text_f, \
            open(wav_scp, 'r', encoding='utf-8') as wav_scp_f, \
            open(utt2spk, 'r', encoding='utf-8') as utt2spk_f, \
            open(segments, 'r', encoding='utf-8') as segments_f, \
            open(reco2file_and_channel, 'r', encoding='utf-8') as reco2file_and_channel_f:

        for line in text_f:
            utt_id = line.split(" ")[0]
            text = " ".join(line.split(" ")[1:]).strip()

            text_dict[utt_id] = text

        for line in wav_scp_f:
            reco_id = line.split(" ")[0]
            for x in line.split(" ")[1:]:
                if "/exp/scale23/data" in x:
                    wav = x.strip()
                    break
            wav_dict[reco_id] = wav

        for line in utt2spk_f:
            utt_id = line.split(" ")[0]
            spk = " ".join(line.split(" ")[1:]).strip()

            utt2spk_dict[utt_id] = spk

        for line in segments_f:
            utt_id = line.split(" ")[0]
            reco_id = line.split(" ")[1]
            start = line.split(" ")[2]
            end = line.split(" ")[3].strip()

            segments_dict[utt_id] = [reco_id, start, end]

        for line in reco2file_and_channel_f:
            reco_id = line.split(" ")[0]
            channel = " ".join(line.split(" ")[2]).strip()

            reco2file_and_channel_dict[reco_id] = channel


    for utt_id in text_dict:
        reco_id = segments_dict[utt_id][0]

        audio_path = wav_dict[reco_id]
        channel = reco2file_and_channel_dict[reco_id]
        spk = utt2spk_dict[utt_id]
        start = segments_dict[utt_id][1]
        end = segments_dict[utt_id][2]
        text = text_dict[utt_id]

        stm_line = "%s %s %s %s %s <O> %s" % (audio_path, channel, spk, start, end, text)

        stm_list.append(stm_line)

    return stm_list


def main(args):
    text = args.text
    wav_scp = args.wav_scp
    utt2spk = args.utt2spk
    segments = args.segments
    reco2file_and_channel = args.reco2file_and_channel
    output_stm = args.output_stm

    stm_list = create_stm(text, wav_scp, utt2spk, segments, reco2file_and_channel)

    with open(output_stm, 'w', encoding='utf-8') as f:
        for stm_line in stm_list:
            f.write("%s\n" % stm_line)


if __name__ == "__main__":
    args = _get_parser().parse_args()

    main(args)