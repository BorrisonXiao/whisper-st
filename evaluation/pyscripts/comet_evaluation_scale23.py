import argparse
from comet import load_from_checkpoint

def read_data(src_file, ref_file, hyp_file, ids_file):
    with open(src_file) as srclines, open(ref_file) as reflines, open(hyp_file) as hyplines, open(ids_file) as idlines:
        testdata = []
        segids = []
        for (src, ref, hyp, ids) in zip(srclines, reflines, hyplines, idlines):
            if src_file.split('.')[-1] == "stm":
                src = " ".join(src.split(' ')[6:])
            if ref_file.split('.')[-1] == "stm":
                ref = " ".join(ref.split(' ')[6:])
            if hyp_file.split('.')[-1] == "stm":
                hyp = " ".join(hyp.split(' ')[6:])
            segid = " ".join(ids.split(' ')[:6])
            testdata.append({"src": src.strip(), "ref":ref.strip(), "mt":hyp.strip()})
            segids.append(segid)
        return testdata, segids

def print_result(model_output, data, segments, system_only=False):
    if not system_only:
        for (seg, score) in zip(segments, model_output['scores']):
            #print(f"{i} src: {data[i]['src']}\tref: {data[i]['ref']}\tmt: {data[i]['mt']}\t{score}")
            print(f"{seg} {score}")
    print(f"System score: {model_output['system_score']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=False, default="../../comet_models/comet/checkpoints/model.ckpt", help="Comet model checkpoint to be used (e.g., ../../comet_models/comet/checkpoints/model.ckpt")
    parser.add_argument("--source", "-s", type=str, required=True, help="Source transcript (usually reference transcription, not ASR output)")
    parser.add_argument("--reference", "-r", type=str, required=True, help="Reference translation")
    parser.add_argument("--target", "-t", type=str, required=True, help="Hypothesis translation to be evaluated")
    parser.add_argument("--ids", "-i", type=str, required=True, help="stm file containing segment ids to go with segment scores")
    parser.add_argument("--only_system", action='store_true', required=False, help="Prints only the final system score. (default: False)")
    args = parser.parse_args()

    print(f"Evaluating {args.target} against source {args.source} and reference {args.reference}")

    model = load_from_checkpoint(args.model)
    data, segs = read_data(args.source, args.reference, args.target, args.ids)
    model_output = model.predict(data, batch_size=8, gpus=1)
    print_result(model_output, data, segs, system_only=args.only_system)
    
    
if __name__ == "__main__":
    main()
