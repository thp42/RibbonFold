# RibbonFold

## 1. Environment Setup

To successfully run the RibbonFold project, you need to set up a Python environment that meets the following requirements. You can either install the environment through the environment.yml file or follow the steps below:

### Create and Activate Python Environment

First, create a new Anaconda environment and activate it:

```bash
conda create -n ribbon_env python=3.9
conda activate ribbon_env
```

### Install packages

Install the following dependencies:

```bash
conda install -y cudatoolkit=11.8 -c nvidia
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchtyping==0.1.4 functorch tensorflow-cpu==2.6.0 tensorflow-estimator==2.14.0
pip install pandas==1.3.5 scipy==1.5.4 biopython dm-tree treelib tqdm ml_collections pytz python-dateutil contextlib2 PyYAML --no-deps
```


### Download model weights
Download the model weights from https://zenodo.org/records/15128410 and unzip the file into ./ckpt folder:

```bash
tar -xzvf model_checkpoints.tar.gz
```


## 2. Inference

To run RibbonFold inference, you need to prepare a MSA file for your input sequence (monomer) first. 

Use the following script to preprocess the MSA features from an AlphaFold2 msa file. A pkl.gz file will be generated and this file should be passed through the following inference script. 

```bash
python process_msa_file.py --input_fasta ./examples/5oqv.fasta --msa_file ./examples/5oqv_msa.a3m --output ./examples/5oqv_msa.pkl.gz

### Modify the run_inference.sh script
An example script is as follows

```bash
CHECKPOINT_PATH="./ckpt/model_ckpt_001.pt"
INPUT_PKL_FILE="./examples/5oqv_msa.pkl.gz"
OUTPUT_DIR="./results/"
ROUNDS=10

python inference.py \
  --checkpoint ${CHECKPOINT_PATH} \
  --input_pkl ${INPUT_PKL_FILE} \
  --ribbon_name 5oqv \
  --output_dir ${OUTPUT_DIR} \
  --rounds ${ROUNDS} \
  --use_dropout true \
  --use_init_structure true
```

Then simply run

```bash
bash run_inference.sh
```



