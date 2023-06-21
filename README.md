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
### ASR Results (WER)
| Model      | iwslt.dev (ara) | iwslt.test (ara) | fleurs.test (ara) |
| ----------- | ----------- | ----------- | ----------- |
| large-v2 | 79.1 | 77.2 | 8.9 |

### ST Results (BLEU)
| Model      | iwslt.dev (ara) | iwslt.test (ara) | fleurs.test (ara) |
| ----------- | ----------- | ----------- | ----------- |
| large-v2 | 3.3 | 3.5 | 21.9 |