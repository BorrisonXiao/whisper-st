#!/usr/bin/env python3

import argparse

REF = "/home/hltcoe/cxiao/scale23/evaluation/scores/iwslt22_ara_dev/data/sr.ara-ara.iwslt22.dev.stm"
OUT = "/home/hltcoe/cxiao/scale23/evaluation/scores/iwslt22_ara_dev/data/ref.stm"


def clean(ref, outfile):
    with open(ref, "r", encoding="utf-8") as f, open(outfile, "w", encoding="utf-8") as f2:
        # Remove the --- in each line (and the spaces after it)
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        for line in lines:
            splitted = line.split(maxsplit=5)
            headers = " ".join(splitted[:5])
            text = splitted[-1]
            text = text.strip()
            text = text.replace(".", "")
            text = text.replace("--", "")
            text = text.replace("!", "")
            text = text.replace("?", "")
            text = text.replace("~", "")
            text = text.replace("  ", " ")
            print(headers, text, file=f2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The raw reference file", default=REF)
    parser.add_argument("-o", "--output", help="The output file", default=OUT)
    args = parser.parse_args()
    clean(ref=args.input, outfile=args.output)


if __name__ == "__main__":
    main()
