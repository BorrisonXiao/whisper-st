# Distributed Whisper Finetuning/Inference

This is a tool for preprocessing data using ESPNet and running distributed whisper inference and finetuning on the COE grid.

# Installation

This project requires the installation of [ESPNet](https://espnet.github.io/espnet/installation.html) with [Kaldi](https://www.notion.so/How-to-Install-Kaldi-on-CLSP-Grid-c9a371dec29248fab99900ef8b453cbc).

After that, clone this repository to a directory of your choice (doesn't have to be under the ESPNet root directory).

Specify the root directory of ESPNet in the `.bashrc` file, e.g.

``export ESPNET_ROOT=/path/to/ESPNet``

Note that the evaluation recipes are modified slightly to handle different (e.g. OOD) datasets and ASR evaluation.

# Python Environment Setup
The ESPNet installation will create a conda environment by default. However, to avoid overwriting ESPNet packages, we need to create a new environment and install the `requirements.txt` there, which includes the dependencies for finetuning models under the huggingface framework.
- Step 1: Create a new conda environment (the name of the environment is set to `cxiao` for disambiguation):

        conda create -n cxiao python=3.9

- Step 2: Activate the environment and install the packages in `requirements.txt`:

        conda activate cxiao; pip install -r requirements.txt

- Step 3: Configure the script to activate the correct conda environment. Open `utils/activate_hf.sh`, change line 6 from

        . /home/hltcoe/cxiao/research/espnet-st/tools/miniconda/etc/profile.d/conda.sh
    to

        . </path/to/your/conda/etc>/profile.d/conda.sh

# Kaldi Installation Guide on the COE Grid
- Step 1: Clone [kaldi](https://github.com/kaldi-asr/kaldi.git) from github.
- Step 2: Load the necessary modules on the Grid:

        ml load intel/mkl/64/2019/5.281
        ml load gcc/7.2.0
- Step 3: Go to the `tools` subdirectory:

        cd tools

- Step 4: Change line 95 of the `Makefile` from

        $(WGET) -nv -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
    to
    
        $(WGET) --no-check-certificate -nv -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
    to ignore the certificate check for `www.openfst.org`.

- Step 5: Run the installation:

        make -j 8

- Step 6: Go to the `src` subdirectory under the kaldi root directory:

        cd ../src

- Step 7: Run the configuration using the loaded `mkl` module:

        ./configure --shared --mkl-root=/cm/shared/apps/intel/compilers_and_libraries/2019.5.281/linux/mkl

- Step 8: Run the final installation step:

        make -j clean; make -j 8

- Step 9: Link the installed kaldi to ESPNet's `tools` directory. To run this, first go to the ESPNet's `tools` directory, e.g. `cd espnet/tools`, and simply run the command:

        ln -s /path/to/kaldi .

# Usage (Evaluation)

The script `run.sh` is an entry point of the decoding pipeline.

One can directly run `./run.sh` on the COE grid to perform data preprocessing, or alternatively, it is recommended to specify the steps in `run.sh` to run one step at a time. It is also recommended to skip some unnecessary steps (e.g. data preparation) to save some time and redundant duplication of the data.

(Theoretically) to skip the first 4 steps, one only needs to run essentially two commands, namely

``ln -sfv /home/hltcoe/cxiao/scale23/whisper/recipe/st/dump .``

and

``ln -sfv /home/hltcoe/cxiao/scale23/whisper/recipe/st/data .``

Please let me (Cihan) know if there is anything missing or there is any bug/error.

# Usage (Finetuning and Evaluation)
The script `run_finetune.sh` and `r*.sh` are the entry points for whisper finetuning.

Stage 1-6 of the script performs data preprocessing (similar to `decode.sh`). It has the additional steps for data stitching (the `merge_utt` flag), which re-merges the utterances in each conversation up to the closest 30s (across speakers).

Stage 7 launches the finetuning job with configurations specified in files under `conf/tuning`.

Stage 8 and 9 runs inference for the finetuned model and evaluates the results.

Before running the script, a few variables should be set explicitly:
- `python_hf=/path/to/your/python`
    
    This variable should point to the python interpreter that has the `requirements.txt` installed.
- `conf/tuning/*.yaml`

    If you are trying to launch a new experiment (with new language, peft method, etc.), you need to create a new configuration file. The naming convention is `conf/tuning/${mode}_${model}_${src_lang}_${peft_method}_${train_set}.yaml` as specified in the `run_finetune.sh` script.

# Known Issues
1. There is currently a gap between step 6 and step 7, which depends on Deb's script for creating huggingface datasets from stm files. However, since all scale data was already converted into huggingface datasets, step 1-6 no longer needs to be re-run.
2. You might run into permission issues when running experiments on a new language or training set. This is becase the features weren't created previously and the training script will extract the features and store them to the merged database. The current solution is to contact me (Cihan) to extract the features for you. 

# Results (Huggingface)

### ASR Results (WER)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 142.3 | - | 14.9 |
| Korean (uhara) |  59.1 | - | 15.6 |
| Chinese (bbn_cts_bolt) | 34.4 | - | 5.1 |
| Spanish (fisher \| callhome) | 32.7 | - \| - | 15.3 |
| Russian (uhara) | 35.8 | - | 6.6 |

### ST Results (BLEU)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 2.5 | - | 22.6 |
| Korean (uhara) | 11.7 | - | 20.7 |
| Chinese (bbn_cts_bolt) |  9.7 | - | 17.4 |
| Spanish (fisher \| callhome) | 30.6 | - \| - | 23.4 |
| Russian (uhara) | 20.2 | - | 28.2 |

# Results (OpenAI)
The results below are evaluated based on the `large-v2` whisper model.
### ASR Results (CER)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 79.1 | 77.2 | 8.9 |
| Korean (uhara) | - | 37.7 | 5.2 |
| Chinese (bbn_cts_bolt) | 39.4 | 36.5 | 23.7 |
| Spanish (fisher \| callhome) | - | - \| - | - |
| Russian (uhara) | - | - | - |

Note that the results for Chinese is unnormalized, i.e. the hyp contains a lot of traditional Chinese characters whereas the ref is in simplified Chinese only. Also the fleurs testset for Arabic is MSA whereas the iwslt dev and test sets are in Tunisian (a dialect).

### ASR Results (WER)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 110.9 | 113.4 | 25.3 |
| Korean (uhara) |  57.3 | 56.7 | 19.5 |
| Chinese (bbn_cts_bolt) | 39.4 | 36.5 | 23.7 |
| Spanish (fisher \| callhome) | 29.1 | 23.4 \| 29.4 | 15.2 |
| Russian (uhara) | 27.3 | 32.8 | 5.5 |

### ST Results (BLEU)
| Language | dev | test | fleurs.test |
| ----------- | ----------- | ----------- | ----------- |
| Arabic (iwslt) | 3.3 | 3.5 | 21.9 |
| Korean (uhara) | 11.2 | 10.2 | 19.9 |
| Chinese (bbn_cts_bolt) |  9.9 | 11.4 | 17.0 |
| Spanish (fisher \| callhome) | 30.3 | 30.1 \| 24.9 | 22.9 |
| Russian (uhara) | 21.5 | 22.2 | 27.6 |