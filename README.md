# Distributed Whisper Finetuning/Inference

This is a tool for preprocessing data using ESPNet and running distributed whisper inference on the COE grid.

# Installation

This project requires the installation [ESPNet](https://espnet.github.io/espnet/installation.html).

After that, clone this repository to a directory of your choice (doesn't have to be under the ESPNet root directory).

Specify the root directory of ESPNet in the `.bashrc` file, e.g.

``export ESPNET_ROOT=/path/to/ESPNet``

Note that the evaluation recipes are modified slightly to handle different (e.g. OOD) datasets and ASR evaluation.

# Usage

The script `run.sh` is an entry point of the pipeline.

One can directly run `./run.sh` on the COE grid to perform data preprocessing, or alternatively, it is recommended to specify the steps in `run.sh` to run one step at a time. It is also recommended to skip some unnecessary steps (e.g. data preparation) to save some time and redundant duplication of the data.

(Theoretically) to skip the first 4 steps, one only needs to run essentially two commands, namely

``ln -sfv /home/hltcoe/cxiao/scale23/whisper/recipe/st/dump .``

and

``ln -sfv /home/hltcoe/cxiao/scale23/whisper/recipe/st/data .``

Please let me (Cihan) know if there is anything missing or there is any bug/error.

# Results
The results below are evaluated based on the `large-v2` whisper model.
### ASR Results (CER)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 79.1 | 77.2 | 8.9 |
| Korean (uhara) | - | 37.7 | 5.2 |
| Chinese (bbn_cts_bolt) | 39.4 | 36.5 | 23.7 |

Note that the results for Chinese is unnormalized, i.e. the hyp contains a lot of traditional Chinese characters whereas the ref is in simplified Chinese only. Also the fleurs testset for Arabic is MSA whereas the iwslt dev and test sets are in Tunisian (a dialect).

### ASR Results (WER)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 110.9 | 113.4 | 25.3 |
| Korean (uhara) |  57.3 | 56.7 | 19.5 |
| Chinese (bbn_cts_bolt) |  - | - | - |

### ST Results (BLEU)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 3.3 | 3.5 | 21.9 |
| Korean (uhara) | 11.2 | 10.2 | 19.9 |
| Chinese (bbn_cts_bolt) |  9.9 | 11.4 | 17.0 |