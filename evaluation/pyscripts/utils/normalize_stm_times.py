import argparse

from collections import defaultdict
import os

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_stm")
    parser.add_argument("output_stm")

    return parser


def main(args):
    input_stm = args.input_stm
    output_stm = args.output_stm

    with open(input_stm, 'r', encoding="utf-8") as in_f, open(output_stm, 'w', encoding="utf-8") as out_f:
        for line in in_f:
            pre_line = " ".join(line.split(" ")[0:3])
            try:
                start = float(line.split(" ")[3])
            except:
                print(input_stm)
                print(line)
                raise
            end = float(line.split(" ")[4])
            post_line = " ".join(line.split(" ")[5:]).strip()

            stm_line = "%s %.2f %.2f %s" % (pre_line, start, end, post_line)

            out_f.write("%s\n" % stm_line)


if __name__ == "__main__":
    args = _get_parser().parse_args()

    main(args)