#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from cclmoff_modules import CCLMoff
from utils import *
import os
import pickle
import torch 
import joblib

__version__ = 'v1.0'
pwd = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(pwd, 'model/cclmoff_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model structure
try:
    model = ProtRNA()
    model = model.to(device)
except Exception as e:
    model = None
    print(f"Warning: Could not initialize ProtRNA model. Details: {e}")

# Load weights if available
weights_loaded = False
if model is not None:
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            weights_loaded = True
        except Exception as e:
            try:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                weights_loaded = True
            except Exception as ex:
                print(f"Warning: Failed to load model weights from {model_path}. Error: {ex}")
    else:
        print(f"Warning: Model weight file not found at {model_path}.")
        print("Please run train_cclmoff.py first to train the model, calibrate predictions, and save model/ files.")

GENOME = os.path.join(pwd, 'data/hg38.fa')

# Load bins/weights and scaler if available
bins = None
weights = None
scaler_instance = None

if os.path.exists(os.path.join(pwd, 'model/bin_weights.pkl')):
    try:
        with open(os.path.join(pwd, 'model/bin_weights.pkl'), 'rb') as f:
            bins, weights = pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load bin weights. Error: {e}")

if os.path.exists(os.path.join(pwd, 'model/minmax_scaler.pkl')):
    try:
        scaler_instance = joblib.load(os.path.join(pwd, 'model/minmax_scaler.pkl'))
    except Exception as e:
        print(f"Warning: Failed to load MinMaxScaler. Error: {e}")

cclmoff = None
if model is not None and bins is not None and weights is not None and scaler_instance is not None and weights_loaded:
    cclmoff = CCLMoff(model=model, bins=bins, weights=weights, ref_genome=GENOME, scaler=scaler_instance)

def check_loaded():
    global cclmoff, model, bins, weights, scaler_instance, weights_loaded
    if cclmoff is None:
        # Retry loading files just in case they were generated since startup
        if os.path.exists(os.path.join(pwd, 'model/bin_weights.pkl')) and os.path.exists(os.path.join(pwd, 'model/minmax_scaler.pkl')):
            with open(os.path.join(pwd, 'model/bin_weights.pkl'), 'rb') as f:
                bins, weights = pickle.load(f)
            scaler_instance = joblib.load(os.path.join(pwd, 'model/minmax_scaler.pkl'))
            
            if model is None:
                model = ProtRNA().to(device)
            
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                weights_loaded = True
            except Exception:
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                weights_loaded = True
            
            cclmoff = CCLMoff(model=model, bins=bins, weights=weights, ref_genome=GENOME, scaler=scaler_instance)
        
        if cclmoff is None:
            raise RuntimeError("CCLMoff model or calibration files (cclmoff_model.pt, bin_weights.pkl, minmax_scaler.pkl) are not loaded. "
                               "Please run train_cclmoff.py first to train the model and generate these files.")

def cal_score(sgr, tar, ref_genome=GENOME):
    check_loaded()
    cclmoff.ref_genome = ref_genome
    return cclmoff.single_score_(sgr, tar)

def cal_scores(df_in, On='On', Off='Off', ref_genome=GENOME):
    check_loaded()
    cclmoff.ref_genome = ref_genome
    df_in['CCLMoff_Score'] = cclmoff.score(df_in)
    return df_in

def cal_spec(df_in, On='On', Off='Off', ref_genome=GENOME):
    check_loaded()
    cclmoff.ref_genome = ref_genome
    spec = cclmoff.spec_per_sgRNA(data_df=df_in, On=On, Off=Off)
    return spec

def cal_casoffinder_spec(sgr, tar, ref_genome, mm=6, dev='G0'):
    check_loaded()
    cclmoff.ref_genome = ref_genome
    cclmoff.casoffinder_mm = mm
    spec = cclmoff.CasoffinderSpec_(sgr, tar, mm=mm, dev=dev)
    return spec

def opti(tar, ref_genome, threshold=0.77, mm=6, dev='G0'):
    check_loaded()
    cclmoff.opti_th = threshold
    cclmoff.ref_genome = ref_genome
    csv_df = cclmoff.opti(target=tar, mm=mm, dev=dev)
    return csv_df

# Setup the argument parser
def get_parser():
    parser = argparse.ArgumentParser(description="CCLMoff Suite "+__version__)
    module_input = parser.add_argument_group("# CCLMoff modules")
    module_input.add_argument("method", metavar="<method>", type=str, default='score', 
        help="Select a method to calculate the score(s): \n \
        score: CCLMoff-Score of sgRNA-DNA, requires [--sgr, --tar]; \n \
        scores: Batch calculation of CCLMoff-Score, requires [--csv], options [--on_item, --off_item, --out]; \n \
        spec: Calculate CCLMoff-Spec, on-target sequence must be in the first line, requires [--csv], options [--on_item, --off_item]; \n \
        off_spec: Perform Cas-Offinder search and calculate CCLMoff-Spec, requires [--sgr, --tar, --genome], options [--mm, --dev]; \n \
        opti: CCLMoff-Opti optimization by mutation, requires [--tar, --genome], options [--threshold, --mm, --dev, --out]; \n "
        )
    
    key_settings = parser.add_argument_group('# Key Settings')
    key_settings.add_argument("--sgr", metavar="<seq>", type=str, default=None, 
        help="sgRNA sequence to analyse (23nt, 20nt+PAM)")
    key_settings.add_argument("--tar", metavar="<seq>", type=str, default=None, 
        help="Target DNA sequence to analyse (23nt, 20nt+PAM)")
    key_settings.add_argument("--csv", metavar="<file>", type=str, default=None, 
        help="CSV file containing sgRNA and Target DNA sequences, headers are On and Off, respectively. (spec: On-target sequence must be in the first line)")
    key_settings.add_argument("--txt", metavar="<file>", type=str, default=None, 
        help="TXT file of the designed sgRNAs.")
    key_settings.add_argument("--genome", metavar="<file>", type=str, default=os.path.join(pwd, 'hg38.fa'), 
        help="Path to the file of reference genome. (default: hg38.fa)")

    out_setting = parser.add_argument_group("# Output Settings")
    out_setting.add_argument('--out', metavar="<file>", type=str, default='default', 
        help="Output file name.")

    other_option = parser.add_argument_group("# Other Options")
    other_option.add_argument('--on_item', metavar="<name>", type=str, default='On', 
        help="sgRNA column name of the csv file (default: On)")
    other_option.add_argument('--off_item', metavar="<name>", type=str, default='Off', 
        help="Off-target column name of the csv file (default: Off)")
    other_option.add_argument('--mm', metavar="<number>", type=int, default=6, 
        help="Mismatch tolerance (default: 6)")
    other_option.add_argument('--dev', metavar="<device>", type=str, default='G', 
        help="GPU/CPU device setting, the same as in the CasOffinder (default: G)")
    other_option.add_argument('--threshold', metavar="<float>", type=float, default=0.76, 
        help="The CCLMoff-Score threshold for mutated sgRNAs. (default: 0.76)")

    parser.add_argument('--version', action='version', version='CCLMoff {}'.format(__version__))

    return parser


# MAIN FUNCTION
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Method 1: CCLMoff-Score
    if args.method == 'score':
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt with NGG PAM'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        score = cal_score(args.sgr, args.tar)
        print('CCLMoff-Score: \n' + str(score))
    
    elif args.method == 'scores':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        scores = cal_scores(df_csv, On=args.on_item, Off=args.off_item)
        if args.out == 'default':
            scores.to_csv('CCLMoff-Score_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            scores.to_csv(out_name, index=False)
        print('CCLMoff-Score calculation done')

    elif args.method == 'spec':
        assert os.path.exists(args.csv), '{} file not exists'.format(args.csv)
        df_csv = pd.read_csv(args.csv, header=0, index_col=None)
        spec = cal_spec(df_csv, On=args.on_item, Off=args.off_item)
        print('CCLMoff-Spec: \n' + str(spec))
        
    elif args.method == 'off_spec':
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt with NGG PAM'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)

        spec = cal_casoffinder_spec(args.sgr, args.tar, ref_genome=args.genome, mm=args.mm, dev=args.dev)
        print('CCLMoff-Spec: \n' + str(spec))
        
    elif args.method == 'opti':
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt with NGG PAM'
        assert os.path.exists(args.genome), '{}\n reference genome file not exists'.format(args.genome)

        opti_result = opti(args.tar, ref_genome=args.genome, threshold=args.threshold, mm=args.mm, dev=args.dev)
        if args.out == 'default':
            opti_result.to_csv('CCLMoff-Opti_optimization_results.csv', index=False)
        else:
            if args.out[-3:] == 'csv':
                out_name = args.out
            else:
                out_name = args.out + '.csv'
            opti_result.to_csv(out_name, index=False)
        print('CCLMoff-Opti optimization done')

    else:
        print('Please choose a correct method')

if __name__== "__main__":
    main()
