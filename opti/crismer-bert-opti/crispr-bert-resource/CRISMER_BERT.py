#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np
import sys

pwd = os.path.dirname(os.path.realpath(__file__))
if pwd not in sys.path:
    sys.path.append(pwd)

from crismer_bert_modules import CRISMER_BERT

def main():
    parser = argparse.ArgumentParser(description="CRISMER-BERT Suite v1.0")
    
    # Subcommands / Methods
    parser.add_argument("method", choices=['score', 'scores', 'spec', 'off_spec', 'opti', 'calibrate'],
                        help="Method to run: score, scores, spec, off_spec, opti, calibrate")
    
    # Inputs
    parser.add_argument("--sgr", type=str, default=None, help="sgRNA sequence (23nt, 20nt+PAM)")
    parser.add_argument("--tar", type=str, default=None, help="Target DNA sequence (23nt, 20nt+PAM)")
    parser.add_argument("--csv", type=str, default=None, help="CSV file containing On/Off sequence pairs")
    parser.add_argument("--genome", type=str, default=None, help="Path to reference genome file (default: hg38.fa)")
    
    # Configuration
    parser.add_argument("--scenario", type=str, choices=['ts1', 'ts2', 'ts3'], default='ts1',
                        help="Scenario config: ts1 (mismatch lower), ts2 (mismatch lower), ts3 (bulge upper). Default is ts1.")
    parser.add_argument("--weights", type=str, default=None, help="Path to fine-tuned model weights (.h5)")
    parser.add_argument("--scaler", type=str, default=None, help="Path to MinMaxScaler (.pkl)")
    parser.add_argument("--bin_weights", type=str, default=None, help="Path to bin weights (.pkl)")
    
    # Output Settings
    parser.add_argument('--out', type=str, default='default', help="Output file name/path")
    parser.add_argument('--out_dir', type=str, default='models', help="Output directory for calibration files (default: models)")
    
    # Column mapping settings
    parser.add_argument('--on_item', type=str, default='On', help="Column name for sgRNA in CSV (default: On)")
    parser.add_argument('--off_item', type=str, default='Off', help="Column name for target DNA in CSV (default: Off)")
    parser.add_argument('--label_item', type=str, default='Active', help="Column name for activity label in CSV (default: Active)")
    
    # Cas-OFFinder settings
    parser.add_argument('--mm', type=int, default=6, help="Mismatch tolerance (default: 6)")
    parser.add_argument('--dev', type=str, default='G0', help="GPU/CPU device setting for Cas-OFFinder (default: G0)")
    parser.add_argument('--threshold', type=float, default=0.76, help="CRISMER-Score threshold for mutated sgRNAs (default: 0.76)")
    
    args = parser.parse_args()
    
    # Instantiating CRISMER_BERT
    crismer = CRISMER_BERT(
        scenario=args.scenario,
        weights_path=args.weights,
        bins_weights_path=args.bin_weights,
        scaler_path=args.scaler,
        opti_th=args.threshold,
        ref_genome=args.genome
    )
    
    # Method execution
    if args.method == 'score':
        assert args.sgr and args.tar, 'Both --sgr and --tar are required for score method'
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt'
        score = crismer.single_score_(args.sgr, args.tar)
        print('CRISMER-BERT-Score: \n' + str(score))
        
    elif args.method == 'scores':
        assert args.csv and os.path.exists(args.csv), f'CSV file {args.csv} does not exist'
        df_csv = pd.read_csv(args.csv)
        scores = crismer.score(df_csv)
        df_csv['CRISMER_BERT_Score'] = scores
        out_path = 'CRISMER-BERT-Score_results.csv' if args.out == 'default' else args.out
        df_csv.to_csv(out_path, index=False)
        print(f'CRISMER-BERT-Score calculation done. Saved to {out_path}')
        
    elif args.method == 'spec':
        assert args.csv and os.path.exists(args.csv), f'CSV file {args.csv} does not exist'
        df_csv = pd.read_csv(args.csv)
        spec = crismer.spec_per_sgRNA(data_df=df_csv, On=args.on_item, Off=args.off_item)
        print('CRISMER-BERT-Spec: \n' + str(spec))
        
    elif args.method == 'off_spec':
        assert args.sgr and args.tar, 'Both --sgr and --tar are required for off_spec method'
        assert len(args.sgr) == 23, 'sgRNA sequence must be 23-nt'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt'
        assert args.genome and os.path.exists(args.genome), f'Reference genome {args.genome} does not exist'
        spec = crismer.CasoffinderSpec_(args.sgr, args.tar, mm=args.mm, dev=args.dev)
        print('CRISMER-BERT-Spec: \n' + str(spec))
        
    elif args.method == 'opti':
        assert args.tar, '--tar is required for opti method'
        assert len(args.tar) == 23, 'Target DNA sequence must be 23-nt'
        assert args.genome and os.path.exists(args.genome), f'Reference genome {args.genome} does not exist'
        
        opti_result = crismer.opti(args.tar, mm=args.mm, dev=args.dev)
        out_path = 'CRISMER-BERT-Opti_optimization_results.csv' if args.out == 'default' else args.out
        opti_result.to_csv(out_path, index=False)
        print(f'CRISMER-BERT-Opti optimization done. Results saved to {out_path}')
        
    elif args.method == 'calibrate':
        assert args.csv and os.path.exists(args.csv), f'CSV file {args.csv} does not exist'
        df_csv = pd.read_csv(args.csv)
        crismer.calibrate(df_csv, on_col=args.on_item, off_col=args.off_item, label_col=args.label_item, output_dir=args.out_dir)

if __name__ == "__main__":
    main()
