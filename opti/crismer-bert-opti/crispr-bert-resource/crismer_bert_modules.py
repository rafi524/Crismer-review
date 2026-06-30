import os
import sys
import time
import numpy as np
import pandas as pd

pwd = os.path.dirname(os.path.realpath(__file__))
if pwd not in sys.path:
    sys.path.append(pwd)

class CRISMER_BERT:
    def __init__(self, scenario=None, weights_path=None, bins_weights_path=None, scaler_path=None, opti_th=None, ref_genome=None):
        self.proj_t = time.time()
        
        # Automatically detect files in models directory if paths are not provided
        detected_weights = None
        detected_scaler = None
        detected_bins_weights = None
        detected_scenario = 'ts2'  # Default scenario if none detected
        
        models_dir = os.path.join(pwd, 'models')
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            for file in os.listdir(models_dir):
                filepath = os.path.join(models_dir, file)
                if not os.path.isfile(filepath):
                    continue
                # Identify weights file (.h5)
                if file.endswith('.h5'):
                    detected_weights = filepath
                    if 'ts3' in file.lower():
                        detected_scenario = 'ts3'
                    elif 'ts1' in file.lower():
                        detected_scenario = 'ts1'
                    elif 'ts2' in file.lower():
                        detected_scenario = 'ts2'
                # Identify scaler file (.pkl)
                elif 'scaler' in file.lower() and file.endswith('.pkl'):
                    detected_scaler = filepath
                # Identify bin weights file (.pkl)
                elif 'bin' in file.lower() and file.endswith('.pkl'):
                    detected_bins_weights = filepath

        if not scenario:
            self.scenario = detected_scenario
        else:
            self.scenario = scenario

        if not weights_path:
            weights_path = detected_weights
            if not weights_path:
                # Fallback to root directory
                for sc in ['ts2', 'ts1', 'ts3']:
                    p = os.path.join(pwd, f'crispr_bert_model_{sc}.h5')
                    if os.path.exists(p):
                        weights_path = p
                        if not scenario:
                            self.scenario = sc
                        break
            
        if not scaler_path:
            scaler_path = detected_scaler
            if not scaler_path:
                p = os.path.join(pwd, f'models/minmax_scaler_{self.scenario}.pkl')
                if os.path.exists(p):
                    scaler_path = p
                else:
                    p = os.path.join(pwd, 'models/minmax_scaler.pkl')
                    if os.path.exists(p):
                        scaler_path = p
            
        if not bins_weights_path:
            bins_weights_path = detected_bins_weights
            if not bins_weights_path:
                p = os.path.join(pwd, f'models/bin_weights_{self.scenario}.pkl')
                if os.path.exists(p):
                    bins_weights_path = p
                else:
                    p = os.path.join(pwd, 'models/bin_weights.pkl')
                    if os.path.exists(p):
                        bins_weights_path = p

        if ref_genome is not None:
            self.ref_genome = ref_genome
        else:
            self.ref_genome = os.path.join(pwd, 'data/hg38.fa')
            
        if opti_th is not None:
            self.opti_th = opti_th
        else:
            self.opti_th = 0.76
            
        # Dynamically import Encoder
        if self.scenario == 'ts3':
            import Encoder_ts3 as enc
        else:
            import Encoder as enc
        self.enc_module = enc
        
        # Load BERT model and weights
        from model_ts1 import build_bert
        self.model = build_bert()
        
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            self.model.load_weights(weights_path)
        else:
            print("Warning: No weights loaded. Model has random initialization.")
                
        # Try to load scaler
        if scaler_path and os.path.exists(scaler_path):
            import joblib
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                print(f"Warning: Could not load scaler from {scaler_path}: {e}")
                self.scaler = None
        else:
            self.scaler = None
                
        # Try to load bins and weights
        if bins_weights_path and os.path.exists(bins_weights_path):
            import pickle
            try:
                with open(bins_weights_path, 'rb') as f:
                    self.bins, self.prob_weight = pickle.load(f)
                print(f"Loaded bins and weights from {bins_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load bins/weights: {e}")
                self.bins = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
                self.prob_weight = np.array([0.0, 0.000255, 0.004813, 0.032304, 0.20901, 0.528241, 0.712388, 0.828807, 0.91069, 0.953488, 0.972763, 1.0])
        else:
            self.bins = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
            self.prob_weight = np.array([0.0, 0.000255, 0.004813, 0.032304, 0.20901, 0.528241, 0.712388, 0.828807, 0.91069, 0.953488, 0.972763, 1.0])

    def prepare_bert_inputs(self, scored_df):
        pairs_data = []
        for idx, row in scored_df.iterrows():
            on_seq = str(row['On'])
            off_seq = str(row['Off'])
            
            if self.scenario == 'ts3':
                # Bulge/indel encoding (Scenario 3)
                on_seq = on_seq.upper().replace('-', '_')
                off_seq = off_seq.upper().replace('-', '_')
                token_list = []
                for k in range(min(len(on_seq), len(off_seq))):
                    pair = on_seq[k] + off_seq[k]
                    if pair == '__' or pair == '--':
                        token_list.append('--')
                    else:
                        token_list.append(pair)
                while len(token_list) < 24:
                    token_list.append('--')
            else:
                # Mismatch-only encoding (Scenario 1 & 2)
                on_seq = on_seq.lower().replace('n', 'x')
                off_seq = off_seq.lower().replace('n', 'x')
                token_list = [on_seq[k] + off_seq[k] for k in range(min(len(on_seq), len(off_seq)))]
                while len(token_list) < 24:
                    token_list.append('xx')
            
            combined_str = " ".join(token_list)
            pairs_data.append([combined_str, 0]) # 0 is dummy label
            
        # Get BERT token indices and segment indices
        X1_input, X2_input = self.enc_module.BERT_encode(pairs_data)
        
        # Get C_RNN one-hot encoding
        df_pairs = pd.DataFrame(pairs_data)
        X_input = np.array(self.enc_module.C_RNN_encode(df_pairs))
        
        return np.array(X_input), np.array(X1_input), np.array(X2_input)

    def score(self, df):
        df = df.copy()
        df.reset_index(inplace=True, drop=True)
        if 'On' not in df.columns or 'Off' not in df.columns:
            scored_df = pd.DataFrame({
                'On': df.iloc[:, 0],
                'Off': df.iloc[:, 3]
            })
        else:
            scored_df = df
        
        X_input, X1_input, X2_input = self.prepare_bert_inputs(scored_df)
        y_pred = self.model.predict([X_input, X1_input, X2_input], batch_size=256, verbose=0)
        y_prob = y_pred[:, 1]
        
        if self.scaler is not None:
            y_prob = self.scaler.transform(y_prob.reshape(-1, 1)).flatten()
            
        return y_prob

    def single_score_(self, new_sgrna, target):
        df = pd.DataFrame({'On': [new_sgrna], 'Off': [target]})
        return self.score(df)[0]

    def score_bin_(self, y_pred):
        if y_pred.shape[0] != 0:
            y_df = pd.DataFrame(y_pred.reshape(-1, 1), columns=['CRISMER-Score'])
            return y_df['CRISMER-Score'].value_counts(bins=self.bins, sort=False).values
        return np.array([0] * (len(self.bins) - 1))

    def single_aggre_(self, y_pred, out_cnt=True):
        cnt = self.score_bin_(y_pred)
        aggre = (cnt * self.prob_weight).sum()
        return np.append(cnt[-4:], aggre) if out_cnt else aggre

    def single_spec_(self, y_pred):
        aggre = self.single_aggre_(y_pred, out_cnt=False)
        return 200 / (200 + aggre)

    def spec_per_sgRNA(self, data_path=None, data_df=None, On='On', Off='Off', target=None, out_df=False):
        if data_df is not None:
            data_set = data_df.copy()
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        
        offt = data_set.loc[:, Off].values
        if target is not None:
            assert len(target) == 23, 'target sequence must have 23 nt'
            # Ensure sequence match is case-insensitive
            offt_upper = np.array([str(o).upper() for o in offt])
            target_upper = target.upper()
            assert np.sum(offt_upper == target_upper) > 0, f'No sequence matches target: {target}'
            
            target_idx = np.where(offt_upper == target_upper)[0][0]
            matched_row = data_set.iloc[[target_idx]]
            other_rows = data_set.iloc[np.delete(np.arange(len(data_set)), target_idx)]
            data_set = pd.concat([matched_row, other_rows])
            
        y_pred = self.score(data_set)
        spec = self.single_spec_(y_pred[1:])
        
        if out_df:
            data_set['CRISMER-Score'] = y_pred
            return spec, data_set
        else:
            return spec

    def CasoffinderSpec_(self, sgrna, target, out_df=False, offtar_search=None, mm=6, dev='G0'):
        if offtar_search is None:
            offtar_search = os.path.join(pwd, 'script/casoffinder_genome.sh')
            
        temp_out = f'.temp_{self.proj_t}_casoffinder.out'
        temp_in = f'.temp_{self.proj_t}_casoffinder.in'
        
        if os.path.exists(temp_out):
            os.remove(temp_out)
        if os.path.exists(temp_in):
            os.remove(temp_in)
            
        os.system("{} {} {} {} {} {}".format(offtar_search, sgrna[:20], self.ref_genome, mm, dev, self.proj_t))
        
        if not os.path.exists(temp_out) or os.path.getsize(temp_out) == 0:
            data_set = pd.DataFrame([['dummy_chr', np.nan, np.nan, target, np.nan, np.nan]])
        else:
            try:
                data_set = pd.read_csv(temp_out, sep="\t", header=None, index_col=None)
            except pd.errors.EmptyDataError:
                data_set = pd.DataFrame([['dummy_chr', np.nan, np.nan, target, np.nan, np.nan]])
                
        offt = data_set.loc[:, 3].values
        offt = np.array([str(t).upper() for t in offt])
        data_set.loc[:, 3] = offt
        data_set = data_set[-data_set[3].str.contains('N|R|W|M|V|Y|K|D|S|J')]
        data_set.drop_duplicates([1, 2, 3], inplace=True)
        
        target_upper = target[:20].upper()
        if data_set[data_set[3].str.contains(target_upper)].shape[0] == 0:
            data_set = pd.concat([data_set, pd.DataFrame([[data_set.iloc[0, 0], np.nan, np.nan, target, np.nan, np.nan]])], ignore_index=True)
            
        data_set = pd.concat([
            data_set[data_set[3].str.contains(target_upper)],
            data_set[-data_set[3].str.contains(target_upper)]
        ])
        
        if out_df:
            spec, out_dset = self.spec_per_sgRNA(data_df=data_set, On=0, Off=3, target=target, out_df=out_df)
        else:
            spec = self.spec_per_sgRNA(data_df=data_set, On=0, Off=3, target=target)
            
        if os.path.exists(temp_out):
            os.remove(temp_out)
            
        if out_df:
            return spec, out_dset
        else:
            return spec

    def opti(self, target, opti_type=None, ref=None, offtar_search=None, mm=6, dev='G0'):
        if opti_type is None:
            opti_pos = []
            opti_seq = []
            for p in range(1, 21):
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif isinstance(opti_type, tuple):
            opti_pos = []
            opti_seq = []
            for p in range(opti_type[0], opti_type[1]):
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif isinstance(opti_type, (np.ndarray, list, range)):
            opti_pos = []
            opti_seq = []
            for p in opti_type:
                opti_pos = np.append(opti_pos, np.array([p, p, p, p], dtype='int'))
                opti_seq = np.append(opti_seq, ['A', 'C', 'G', 'T'])
            opti_nt = pd.DataFrame()
            opti_nt['Pos'] = opti_pos
            opti_nt['nt'] = opti_seq
        elif isinstance(opti_type, pd.DataFrame):
            opti_nt = opti_type
        else:
            opti_nt = pd.read_csv(opti_type, header=0, index_col=0)

        results = []
        
        # Calculate WT specificity score
        spec, out_df = self.CasoffinderSpec_(target, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
        
        cnt_results = self.score_bin_(out_df.loc[:, 'CRISMER-Score'].values[1:])[-8:]
        cnt_results = np.append(cnt_results[:3], cnt_results[3:].sum())
        
        if ref is not None:
            ref_merge = ref.loc[:, ['Offtarget_Sequence', 'GUIDE-Seq Reads']]
            ref_merge.columns = [3, 'Reads']
            out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
            out_merge.fillna(0, inplace=True)
            out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
            if out_merge.shape[0] != 0:
                out_merge.to_csv('{}_ref_match.tsv'.format(target), sep='\t')
                spec_ref = self.single_spec_(out_merge.loc[:, 'CRISMER-Score'].values[1:], out_cnt=False)
            else:
                spec_ref = 1.0
            results.append([target, target, 'WT', out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec, spec_ref])
        else:
            results.append([target, target, 'WT', out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec])
            
        # Calculate specificity scores of modified sgRNAs
        for i in range(opti_nt.shape[0]):
            pos = int(opti_nt.loc[i, 'Pos'])
            nt = opti_nt.loc[i, 'nt']
            if target[pos - 1] != nt:
                new_sgrna = target[:pos - 1] + nt + target[pos:]
                single_score = self.single_score_(new_sgrna, target)
                if single_score >= self.opti_th:
                    spec, out_df = self.CasoffinderSpec_(new_sgrna, target, out_df=True, offtar_search=offtar_search, mm=mm, dev=dev)
                    cnt_results = self.score_bin_(out_df.loc[:, 'CRISMER-Score'].values[1:])[-5:]
                    cnt_results = np.append(cnt_results[:3], cnt_results[3:].sum())
                    
                    if ref is not None:
                        out_merge = pd.merge(out_df.iloc[1:,:], ref_merge, how='left', on=[3], sort=False)
                        out_merge.fillna(0, inplace=True)
                        out_merge = out_merge.loc[out_merge.loc[:,'Reads'] > 0, :]
                        if out_merge.shape[0] != 0:
                            spec_ref = self.single_spec_(out_merge.loc[:, 'CRISMER-Score'].values[1:], out_cnt=False)
                        else:
                            spec_ref = 1.0
                            
                    if ref is not None:
                        results.append([new_sgrna, target, '{}{}>{}'.format(target[pos - 1], pos, nt), out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec, spec_ref])
                    else:
                        results.append([new_sgrna, target, '{}{}>{}'.format(target[pos - 1], pos, nt), out_df.loc[:, 'CRISMER-Score'].values[0]] + list(cnt_results) + [spec])

        results = np.array(results)
        if results.shape[0] == 1:
            results = np.vstack([results, np.array([[np.nan for _ in range(results.shape[1])]], dtype=object)])
            results[1, 1:3] = np.array([target, 'Optimization unavailable'])
        
        if ref is not None:
            results_df = pd.DataFrame(results, columns=['sgRNA', 'Target', 'Mutation', 'CRISMER-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                        '(0.75, 0.8]', '(0.8, 1.0]', 'CRISMER-Spec', 'ref_Spec'])
        else:
            results_df = pd.DataFrame(results, columns=['sgRNA', 'Target', 'Mutation', 'CRISMER-Score', '(0.65, 0.7]', '(0.7, 0.75]', 
                                                        '(0.75, 0.8]', '(0.8, 1.0]', 'CRISMER-Spec'])
                                                        
        results_df.iloc[:, 3:] = np.array(results_df.iloc[:, 3:].values, dtype='float')
        try:
            results_df.iloc[:, 4:8] = np.array(results_df.iloc[:, 4:8].values, dtype='int')
        except Exception:
            results_df.iloc[0, 4:8] = np.array(results_df.iloc[0, 4:8].values, dtype='int')
            
        results_df['delta_Spec'] = results_df.loc[:, 'CRISMER-Spec'].values - results_df.loc[0, 'CRISMER-Spec']
        results_df['rk'] = np.array([0] + [1] * (results_df.shape[0]-1))

        results_out = results_df.sort_values(by=['rk', 'delta_Spec'], ascending=[True, False])
        results_out.index = np.arange(results_out.shape[0])
        
        return results_out.iloc[:, :-1]

    def calibrate(self, df, on_col, off_col, label_col, output_dir):
        import joblib
        import pickle
        from sklearn.preprocessing import MinMaxScaler
        
        print("Starting calibration of model predictions...")
        # 1. Run predictions
        y_prob = self.score(df)
        labels = df[label_col].values.astype(int)
        
        # 2. Fit MinMaxScaler
        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(y_prob.reshape(-1, 1)).flatten()
        
        os.makedirs(output_dir, exist_ok=True)
        scaler_path = os.path.join(output_dir, f'minmax_scaler_{self.scenario}.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler calibrated and saved to {scaler_path}")
        
        # 3. Calculate bin weights
        bins = [0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
        
        data = pd.DataFrame({
            'score': scaled_scores,
            'Active': labels
        })
        data['bin'] = pd.cut(data['score'], bins=bins, include_lowest=True)
        
        bin_stats = data.groupby('bin', observed=False).apply(
            lambda x: pd.Series({
                'active_count': (x['Active'] == 1).sum(),
                'total_count': len(x)
            }),
            include_groups=False
        ).reset_index()
        
        bin_stats['active_ratio'] = bin_stats['active_count'] / bin_stats['total_count']
        bin_stats['active_ratio'] = bin_stats['active_ratio'].fillna(0)
        
        weights = bin_stats['active_ratio'].values
        
        weights_path = os.path.join(output_dir, f'bin_weights_{self.scenario}.pkl')
        with open(weights_path, 'wb') as f:
            pickle.dump((bins, bin_stats['active_ratio']), f)
            
        print(f"Bin weights calibrated and saved to {weights_path}")
        print("Calibrated Active Ratios per Bin:")
        print(bin_stats)
        
        # Update calibration parameters in memory
        self.scaler = scaler
        self.bins = bins
        self.prob_weight = weights
