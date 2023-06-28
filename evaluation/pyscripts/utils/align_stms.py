import argparse

from collections import defaultdict


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp_stm")
    parser.add_argument("ref_stm")
    parser.add_argument("output_stm")

    return parser


def main(args):
    hyp_stm = args.hyp_stm
    ref_stm = args.ref_stm
    output_stm = args.output_stm

    hyp_dict = {}
    ref_list = []

    stm_lists = []

    with open(hyp_stm, 'r', encoding='utf-8') as hyp_f, \
         open(ref_stm, 'r', encoding='utf-8') as ref_f, \
         open(output_stm, 'w', encoding='utf-8') as out_f:

        for line in hyp_f:
            key = " ".join(line.split(" ")[0:6])
            text = " ".join(line.split(" ")[6:]).strip()

            hyp_dict[key] = text

        for line in ref_f:
            key = " ".join(line.split(" ")[0:6])
            ref_list.append(key)

        for key in ref_list:
            if key in hyp_dict:
                # If line exists in hyp, re-add as-is
                stm_line = "%s %s" % (key, hyp_dict[key])
                stm_lists.append(stm_line)
            else:
                # If line doesn't exist in hyp, add as empty line
                stm_line = "%s " % key
                stm_lists.append(stm_line)

        for stm_line in stm_lists:
            out_f.write("%s\n" % stm_line)

    if len(hyp_dict) < len(stm_lists):
        print("Empty lines were re-added. Total lines added: %d" % (len(stm_lists) - len(hyp_dict)))
    elif len(hyp_dict) == len(stm_lists):
        print("No new lines added")
    else:
        print("Removed extra lines not in reference. Total lines removed: %d" % (len(hyp_dict) - len(stm_lists)))


if __name__ == "__main__":
    args = _get_parser().parse_args()

    main(args)