# CCLMoff-opti

CCLMoff-opti is a tool for CRISPR guide RNA design and optimization utilizing a fine-tuned RNA-FM sequence representation model (ProtRNA).

## Dependencies

### Python Packages
- pandas 
- numpy
- torch 
- scikit-learn 
- scipy 
- joblib
- rna-fm (Installs the RNA-FM model and alphabet converter)

### External Tools
- Cas-OFFinder==2.4 - For off-target search in genomes

### Data Requirements
- Reference genome file (Can be downloaded from UCSC/rgenome website, hg38 is recommended)
- `all_off_target.csv` dataset (located at the root folder of the project) for training the model.

## Set Up

1. Create a python3 environment (conda is recommended)

2. Install required python libraries:
```bash
# Using conda or pip
conda install pandas numpy scipy joblib scikit-learn
conda install pytorch -c pytorch

# Install the RNA-FM package
pip install rna-fm
```

3. Download and install Cas-OFFinder. NOTE: Properly set the environment variables to make Cas-OFFinder available.
   - Visit: https://github.com/snugel/cas-offinder for installation instructions

4. Download genome reference file from the UCSC website and uncompress it to data/. (e.g., data/hg38.fa).

## Training & Model Calibration

Since the ProtRNA (RNA-FM) model, the MinMaxScaler, and the active ratio bin weights must be calibrated specifically for CCLMoff on the off-target dataset, you need to run the training script first:

```bash
python train_cclmoff.py --epochs 10 --batch_size 128
```

This script will:
1. Programmatically download the pre-trained RNA-FM weights (`RNA-FM_pretrained.pth`) to the torch hub cache directory if they are not already cached.
2. Load `all_off_target.csv` from standard paths.
3. Fine-tune the ProtRNA model for 10 epochs.
4. Save the best fine-tuned model weights to `models/cclmoff_model.pth`.
5. Perform temperature-scaled predictions (T=10) on the entire dataset.
6. Calibrate and save the MinMaxScaler to `models/minmax_scaler.pkl`.
7. Calculate active ratios for predicted score bins and save them to `models/bin_weights.pkl`.
8. Save training and validation performance plots to `cclmoff_model_curves.png` and raw performance values to `cclmoff_model_curves.npz`.

Once training is complete, the `models/` directory will be fully populated and `CCLMoff.py` will be runnable.

## Usage

The tool includes several modules for CRISPR guide RNA analysis:

- **CCLMoff-Opti**: Optimize sgRNAs through strategic mutations to improve specificity while maintaining high activity. This is the primary functionality of the tool.
- **CCLMoff off_spec**: Calculate the genome wide specificity of sgRNAs.
- **CCLMoff-Score**: Calculate the activity score of sgRNA-DNA pairs.
- **CCLMoff-Spec**: Calculate the specificity of given sgRNA.

### Examples

```bash
# Optimize a target sequence for improved specificity
python CCLMoff.py opti --tar CGTGCGCAGGAGGACGAGGACGG --genome hg38.fa --out example/pcsk9_out.csv

# Genome wide specificity
python CCLMoff.py off_spec --sgr GAGTCCGAGCAGAAGAAGAANGG --tar GAGTCCGAGCAGAAGAAGAAGGG --genome data/hg38.fa

# Calculate CCLMoff-Score for a sgRNA-target pair
python CCLMoff.py score --sgr ACGTACGTACGTACGTACGTAGG --tar ACGTACGTACGTACGTACGTAGG

# Calculate scores for multiple sgRNA-target pairs
python CCLMoff.py scores --csv example/example.csv
```
