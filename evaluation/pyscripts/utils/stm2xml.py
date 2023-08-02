#!/usr/bin/env python

import argparse
from pathlib import Path
from local.stm import parse_StmUtterance
import os


def stm2xml(src_lang, tgt_lang, refdir, outdir):
    # Convert the reference stm files to xml files
    langdir = refdir / src_lang

    for dset in ["dev1", "dev2"]:
        for (stm_file, out_name) in [(langdir / f"st.{src_lang}-{tgt_lang}.{dset}.stm", "ref_mt.xml"), (langdir / f"sr.{src_lang}-{src_lang}.{dset}.stm", "ref_sr.xml")]:
            with open(stm_file, "r") as f:
                lines = f.readlines()

            stm_utts = [parse_StmUtterance(line.strip()) for line in lines]
            # The stm file will be sorted by start time and root filename
            stm_utts = sorted(stm_utts, key=lambda x: x.start_time)
            stm_utts = sorted(stm_utts, key=lambda x: Path(x.filename).stem)

            _outdir = outdir / dset
            _outdir.mkdir(parents=True, exist_ok=True)
            with open(_outdir / out_name, "w") as f:
                # First line is something like: <?xml version = "1.0" encoding = "UTF-8"?>
                f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
                # Second line is <mteval>
                f.write("<mteval>\n")
                # Third line is <refset setid="iwslt-ACLdev2023" srclang="English" trglang="Arabic" refid="ref">
                if out_name == "ref_sr.xml":
                    f.write("<srcset setid=\"{}\" srclang=\"{}\">\n".format(
                        "test", src_lang))
                else:
                    f.write("<refset setid=\"{}\" srclang=\"{}\" trglang=\"{}\" refid=\"ref\">\n".format(
                        dset, src_lang, tgt_lang))
                # Add a dummy docid line, e.g. <doc docid="2022.acl-long.268" genre="presentations">
                f.write("<doc docid=\"{}\" genre=\"presentations\">\n".format(dset))
                # Add a dummy talkid line, e.g. <talkid>2022.acl-long.268</talkid>
                f.write("<talkid>{}</talkid>\n".format(dset))
                for utt in stm_utts:
                    # utt = parse_StmUtterance(line)
                    f.write(
                        "<seg id=\"{}\">{}</seg>\n".format(utt.utterance_id(stereo=True), utt.transcript))
                f.write("</doc>\n")
                if out_name == "ref_sr.xml":
                    f.write("</srcset>\n")
                else:
                    f.write("</refset>\n")
                f.write("</mteval>\n")

    for _dir in [langdir / "testsets" / "cts", langdir / "testsets" / "ood"]:
        if _dir == langdir / "testsets" / "ood":
            stm_files = list(_dir.glob(f"sr.{src_lang}-{src_lang}.fleurs.test.stm")) + \
                list(_dir.glob(f"st.{src_lang}-{tgt_lang}.fleurs.test.stm"))
        else:
            stm_files = list(_dir.glob(f"sr.{src_lang}-{src_lang}*.test.stm")) + \
                list(_dir.glob(f"st.{src_lang}-{tgt_lang}*.test.stm"))
        for stm_file in stm_files:
            # dset is the wildcard in the stm file name, e.g. "bbn_cts_bolt"
            dset = str(os.path.basename(stm_file).split(".")[2]) + "_test"
            out_name = "ref_sr.xml" if "sr." in str(stm_file) else "ref_mt.xml"
            with open(stm_file, "r") as f:
                lines = f.readlines()

            stm_utts = [parse_StmUtterance(line.strip()) for line in lines]
            # The stm file will be sorted by start time and root filename
            stm_utts = sorted(stm_utts, key=lambda x: x.start_time)
            stm_utts = sorted(stm_utts, key=lambda x: Path(x.filename).stem)

            _outdir = outdir / dset
            _outdir.mkdir(parents=True, exist_ok=True)
            with open(_outdir / out_name, "w") as f:
                # First line is something like: <?xml version = "1.0" encoding = "UTF-8"?>
                f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
                # Second line is <mteval>
                f.write("<mteval>\n")
                # Third line is <refset setid="iwslt-ACLdev2023" srclang="English" trglang="Arabic" refid="ref">
                if out_name == "ref_sr.xml":
                    f.write("<srcset setid=\"{}\" srclang=\"{}\">\n".format(
                        "test", src_lang))
                else:
                    f.write("<refset setid=\"{}\" srclang=\"{}\" trglang=\"{}\" refid=\"ref\">\n".format(
                        "test", src_lang, tgt_lang))
                f.write("<doc docid=\"{}\" genre=\"presentations\">\n".format(dset))
                # Add a dummy talkid line, e.g. <talkid>2022.acl-long.268</talkid>
                f.write("<talkid>{}</talkid>\n".format(dset))
                for utt in stm_utts:
                    # utt = parse_StmUtterance(line)
                    f.write(
                        "<seg id=\"{}\">{}</seg>\n".format(utt.utterance_id(stereo=True), utt.transcript))
                f.write("</doc>\n")
                if out_name == "ref_sr.xml":
                    f.write("</srcset>\n")
                else:
                    f.write("</refset>\n")
                f.write("</mteval>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, help="Source language")
    parser.add_argument("--tgt_lang", type=str, help="Target language")
    parser.add_argument("--refdir", type=Path,
                        help="Path to the directory containing the stm files")
    parser.add_argument("--outdir", type=Path,
                        help="Path to the output directory")
    args = parser.parse_args()

    stm2xml(src_lang=args.src_lang, tgt_lang=args.tgt_lang,
            refdir=args.refdir, outdir=args.outdir)


if __name__ == "__main__":
    main()
