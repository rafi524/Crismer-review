# DipOff-opti

DipOff-opti is a tool for CRISPR guide RNA design and optimization using a BiLSTM sequence classification architecture.

## Dependencies

### Python Packages
- pandas 
- numpy
- torch 
- scikit-learn 
- scipy 
- joblib

### External Tools
- Cas-OFFinder==2.4 - For off-target search in genomes

### Data Requirements
- Reference genome file (Can be downloaded from UCSC/rgenome website, hg38 is recommended)
- `all_off_target.csv` dataset (located at the root folder `E:/CRISMER-Extended/all_off_target.csv`) for training the model.

## Set Up

1. Create a python3 environment (conda is recommended)

2. Install required python libraries:
```bash
# Using conda
conda install pandas numpy scipy joblib scikit-learn
conda install pytorch -c pytorch

# Alternative: using pip
pip install pandas numpy torch scikit-learn scipy joblib
```

3. Download and install Cas-OFFinder. NOTE: Properly set the environment variables to make Cas-OFFinder available.
   - Visit: https://github.com/snugel/cas-offinder for installation instructions

4. Download genome reference file from the UCSC website and uncompress it to data/. (e.g., data/hg38.fa).

## Training & Model Calibration

Since the BiLSTM model, the MinMaxScaler, and the active ratio bin weights must be calibrated specifically for DIPOFF on the off-target dataset, you need to run the training script first:

```bash
python train_dipoff.py --epochs 50 --batch_size 64
```

This script will:
1. Load `all_off_target.csv` from the root directory.
2. Train the DIPOFF BiLSTM model.
3. Save the model weights to `models/dipoff_lstm_model.pth`.
4. Perform temperature-scaled softmax predictions (T=10) on the dataset.
5. Calibrate and save the MinMaxScaler to `models/minmax_scaler.pkl`.
6. Calculate active ratios for predicted score bins and save them to `models/bin_weights.pkl`.

Once training is complete, the `models/` directory will be fully populated and `DIPOFF.py` will be runnable.

## Usage

The tool includes several modules for CRISPR guide RNA analysis:

- **DIPOFF-Opti**: Optimize sgRNAs through strategic mutations to improve specificity while maintaining high activity. This is the primary functionality of the tool.
- **DIPOFF off_spec**: Calculate the genome wide specificity of sgRNAs.
- **DIPOFF-Score**: Calculate the activity score of sgRNA-DNA pairs.
- **DIPOFF-Spec**: Calculate the specificity of given sgRNA.

### Examples

```bash
# Optimize a target sequence for improved specificity
python DIPOFF.py opti --tar CGTGCGCAGGAGGACGAGGACGG --genome hg38.fa --out example/pcsk9_out.csv

# Genome wide specificity
python DIPOFF.py off_spec --sgr GAGTCCGAGCAGAAGAAGAANGG --tar GAGTCCGAGCAGAAGAAGAAGGG --genome data/hg38.fa

# Calculate DIPOFF-Score for a sgRNA-target pair
python DIPOFF.py score --sgr ACGTACGTACGTACGTACGTAGG --tar ACGTACGTACGTACGTACGTAGG

# Calculate scores for multiple sgRNA-target pairs
python DIPOFF.py scores --csv example/example.csv
```
