# Vast.ai Setup and Experiment Guide

This guide details how to rent, setup, and run the CRISPR-BERT experiments on a **Vast.ai** GPU instance.

---

## 1. Renting a Vast.ai Instance

TensorFlow 2.5.0 requires legacy CUDA and cuDNN versions that do not run out-of-the-box on newer CUDA 12 templates. Follow these steps carefully:

### Selecting the Right Template & Docker Image
1. Go to the **Create** page on Vast.ai.
2. Click **EDIT IMAGE & CONFIG TEMPLATE** (top left).
3. Choose **Ubunutu** / **CUDA** base image.
4. Set the **Docker Image** field exactly to:
   ```bash
   nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
   ```
   > [!IMPORTANT]
   > Do NOT use CUDA 12 templates. TensorFlow 2.5.0 requires CUDA 11.2 and cuDNN 8.1. Newer CUDA versions will fail to register the GPU device.
5. In **Launch Settings**, check **Use Jupyter Lab** (if you want notebook access, though we run raw `.py` scripts) and make sure SSH access is enabled.
6. Allocate at least **20 GB** of disk space.

### Selecting a GPU
- Any modern NVIDIA GPU will work (e.g., RTX 3090, RTX 4090, A100, RTX A4000/A5000). A single GPU is sufficient.

---

## 2. Environment Verification

Once your instance starts, connect via SSH or open the Jupyter terminal, then run:

```bash
# Verify the GPU is detected and check current NVIDIA driver version
nvidia-smi
```

Confirm that the output lists your GPU device.

---

## 3. Creating a Conda Python 3.9 Environment

TensorFlow 2.5.0 requires **Python 3.6 to 3.9**. Since most host templates default to Python 3.10+, you must set up a custom Miniconda environment:

```bash
# 1. Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# 2. Initialize shell config
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init

# 3. Create a Python 3.9 environment
conda create -n crispr_bert python=3.9 -y
conda activate crispr_bert
```

---

## 4. Installing Dependencies

Navigate to your copied `crispr-bert` directory and install the requirements:

```bash
cd new_exp/crispr-bert
pip install -r requirements.txt
```

---

## 5. Running the Experiments

Since these scripts are fully configured for headless execution, they will output training progress to the terminal (and log files) and save the charts directly as PNGs without throwing window-display errors:

```bash
# Run Test Scenario 1 (Trains on all_off_target.csv)
python model_train_ts1.py

# Run Test Scenario 2 (Trains on changeseq_siteseq.csv & tests on 4 datasets)
python model_train_ts2.py

# Run Test Scenario 3 (Trains on I1.txt bulge dataset)
python model_train_ts3.py
```

---

## 6. Verifying the Outputs

After execution completes, you can check that the following outputs are generated inside the folder:

### Console Output Logs (Stdout/Stderr)
- `ts1_output.log` — Full stdout logs for Scenario 1.
- `ts2_output.log` — Full stdout logs for Scenario 2.
- `ts3_output.log` — Full stdout logs for Scenario 3.

### Clean Result Summaries
- `ts1_metrics.txt` — Final standard and bootstrap metrics (ROC AUC, F1-Score, PR AUC with 95% CI) for Scenario 1.
- `ts2_metrics.txt` — Performance summaries and bootstrap intervals for all 4 external test datasets (Circleseq, surroseq, guideseq, ttiss).
- `ts3_metrics.txt` — Final standard and bootstrap metrics for Scenario 3.

### Saved Weights & npz Files
- `crispr_bert_model_ts1.h5` / `crispr_bert_curves_ts1.npz`
- `crispr_bert_model_ts2.h5`
- `crispr_bert_model_ts3.h5` / `crispr_bert_curves_ts3.npz`

### Evaluation Curves (Saved as PNGs)
- `crispr_bert_curves_ts1.png` — Loss convergence, ROC curve, and PR curves.
- `crispr_bert_curves_ts3.png` — Loss convergence, ROC curve, and PR curves.

---

## 7. Using the CRISMER-BERT Suite

A unified command-line tool `CRISMER_BERT.py` is included to run predictions, calculate specificity scores, calibrate outputs, and perform sgRNA specificity optimization.

### Scoring sgRNA-Target Pairs
To calculate the cleavage probability score of a single sgRNA-DNA target pair:
```bash
python CRISMER_BERT.py score --sgr CGTGCGCAGGAGGACGAGGACGG --tar CGTGCGCAGGAGGACGAGGACGG --scenario ts1 --weights crispr_bert_model_ts1.h5
```

### Batch Calculation of Cleavage Scores
To compute scores for multiple sgRNA-target pairs defined in a CSV file:
```bash
python CRISMER_BERT.py scores --csv example.csv --on_item On --off_item Off --out results.csv --scenario ts1 --weights crispr_bert_model_ts1.h5
```

### Specifying Genomic Specificity (CRISMER-Spec)
Calculate specificity score after running a Cas-OFFinder genomic search:
```bash
python CRISMER_BERT.py off_spec --sgr CGTGCGCAGGAGGACGAGGACGG --tar CGTGCGCAGGAGGACGAGGACGG --genome hg38.fa --mm 6 --dev G0 --scenario ts1 --weights crispr_bert_model_ts1.h5
```

### Mutational Optimization (CRISMER-Opti)
Generate mutant sgRNAs with higher specificity (lower off-target potential) for a given target sequence:
```bash
python CRISMER_BERT.py opti --tar CGTGCGCAGGAGGACGAGGACGG --genome hg38.fa --mm 6 --dev G0 --scenario ts1 --weights crispr_bert_model_ts1.h5 --out optimized_sgrnas.csv
```

### Scaler & Weight Calibration
To calibrate the probability outputs and calculate custom active ratios per bin from a dataset:
```bash
python CRISMER_BERT.py calibrate --csv datasets/all_off_target.csv --on_item "Target sgRNA" --off_item "Off Target sgRNA" --label_item "label" --scenario ts1 --weights crispr_bert_model_ts1.h5 --out_dir models
```
After running calibration, the CLI will automatically detect the generated `minmax_scaler_ts1.pkl` and `bin_weights_ts1.pkl` files inside `models/` directory for subsequent scoring and optimization commands.

