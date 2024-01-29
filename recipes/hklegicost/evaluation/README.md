# Evalutation 

The purpose of this repository is to have the evalutations scripts to be used for SCALE2023. 

## Evaluation metrics

The current list of evalutation metrics we will be using are the following:

### Main

- BLEU​
- Translation Error Rate​ (TER)
- chrF
- Word and Character Error Rate (WER and CER)​

### Other Metrics
- COMET​
- Downstream Tasks (e.g. NER, CLIR, etc.)

## Evaluation tools

For the Main metrics we will be using:

- `sacreBLEU` - for BLEU, TER, and chrF
- `sclite` - for WER and CER

For the Other Metrics, evaluations tools are TBD.

## Other Scripts

This repo will also contain scripts to normalize the data into a standard format for the scripts (e.g. lowercase, untokenize, etc.).
