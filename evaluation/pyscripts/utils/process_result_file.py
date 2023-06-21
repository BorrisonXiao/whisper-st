import argparse
import json

def _get_parser():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("results_file", help="The file containing the results")
    parser.add_argument("--wer", action="store_true")

    return parser


def parse_json_array(json_arr):
    bleu = 0
    ter = 0
    chrf2 = 0

    found_extra = False

    for element in json_arr:
        score = element["score"]
        if element["name"] == "BLEU":
            bleu = score
        elif element["name"] == "chrF2":
            chrf2 = score
        elif element["name"] == "TER":
            ter = score
        else:
            found_extra = True
            continue

    if found_extra:
        return False

    return bleu, chrf2, ter


def parse_bleu(results_file):
    # Initialize empty strings for storing json string
    single_ref_arr = ""
    multi_ref_arr = ""

    # Variables used to decide which type of scoring was done
    single_ref = False
    multi_ref = False

    with open(results_file) as f:
        for line in f:
            # Parse through each line of the file to identify json object for each type of result
            if "single-reference" in line:
                single_ref = True
                multi_ref = False
                continue
            if "multi-references" in line:
                single_ref = False
                multi_ref = True
                continue
            if single_ref:
                single_ref_arr += line.strip()
            if multi_ref:
                multi_ref_arr += line.strip()

    if len(single_ref_arr):
        single_ref_json = json.loads(single_ref_arr)
        single_parser_result = parse_json_array(single_ref_json)
    else:
        single_parser_result = False

    if len(multi_ref_arr):
        multi_ref_json = json.loads(multi_ref_arr)
        multi_parser_result = parse_json_array(multi_ref_json)
    else:
        multi_parser_result = False

    if not single_parser_result:
        print("No single reference found")
    else:
        print("Single Reference Score")
        print("BLEU, chrF2, TER")
        print("%s, %s, %s" % single_parser_result)

    if not multi_parser_result:
        print("No multi reference found")
    else:
        print("Multi Reference Score")
        print("BLEU, chrF2, TER")
        print("%s, %s, %s" % multi_parser_result)

    print("\n")


def parse_wer(results_file):

    wer = ''
    with open(results_file) as f:
        for line in f:
            if "Sum/Avg" in line:
                splitted_line = line.split() #" ".join(line.split(" ")).split(" ")
                wer = splitted_line[9]

                break

    print("WER")
    print("%s\n" % wer)


def main(args):
    results_file = args.results_file

    if args.wer:
        parse_wer(results_file)
    else:
        parse_bleu(results_file)



if __name__ == "__main__":

    args = _get_parser().parse_args()

    main(args)
