import numpy as np
import pandas as pd

def one_hot_features(df):
    print("Generating One hot encoding features...")
    
    # Nucleotides and possible pairs
    nucleotides = ['A', 'T', 'G', 'C']
    pairs = [f'{n1}{n2}' for n1 in nucleotides for n2 in nucleotides]  # 16 possible pairs
    
    # Initialize the pairwise feature matrix (rows = positions, columns = 16 pairs)
    pairwise_features = np.zeros((len(df), 20, len(pairs)))  # (samples, positions=20, pairs=16)
    
    # Loop through each row in the DataFrame and populate the pairwise features
    for idx, row in df.iterrows():
        on_seq = row['On']
        off_seq = row['Off']
        
        for pos in range(20):  # Loop through positions 1 to 20
            pair = on_seq[pos] + off_seq[pos]  # Create the pair from the same position in both sequences
            if pair in pairs:
                pair_idx = pairs.index(pair)  # Get the index of the pair
                pairwise_features[idx, pos, pair_idx] = 1  # Set the feature value to 1
    
    # Return a DataFrame with the pairwise features
    # Reshape to (len(df), 20, 16) as the final output
    return pairwise_features

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

def eval_matrices(model, test_x, test_y, debug = True):
    true_y, results = tester(model, test_x, test_y)
    predictions = [torch.nn.functional.softmax(r) for r in results]
    pred_y = np.array([y[1].item() for y in predictions])
    pred_y_list = []
    test_y = np.array([y.item() for y in true_y])

    for x in pred_y:
        if(x>0.5):
            pred_y_list.append(1)
        else:
            pred_y_list.append(0)

    pred_y_list = np.array(pred_y_list)
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y_list).ravel()
    precision, recall, _ = precision_recall_curve(test_y, pred_y)
    auc_score = auc(recall, precision)
    acc = accuracy_score(test_y, pred_y_list)

    pr = -1
    re = -1
    f1 = -1
    try:
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        f1 = 2*pr*re / (pr+re)
    except:
        f1 = -1

    stats = Stats()
    stats.acc = acc
    stats.pre = pr
    stats.re = re
    stats.f1 = f1
    stats.roc = roc_auc_score(test_y, pred_y)
    stats.prc = auc_score
    stats.tn = tn
    stats.fp = fp
    stats.fn = fn
    stats.tp = tp

    if debug:
        print('Accuracy: %.4f' %acc)
        print('Precision: %.4f' %pr)
        print('Recall: %.4f' %re)
        print('F1 Score: %.4f' %f1)
        print('ROC:',roc_auc_score(test_y, pred_y))
        print('PR AUC: %.4f' % auc_score)

        # print(classification_report(test_y, pred_y_list, digits=4))
        # print("Confusion Matrix")
        # print(confusion_matrix(test_y, pred_y_list))

    return stats


def eval(model,test_df):
    
    test_x = one_hot_features(test_df)
    test_y = test_df['Active'].to_numpy()
    stats = eval_matrices(model, test_x, test_y)
