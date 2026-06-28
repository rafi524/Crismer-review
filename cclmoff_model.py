import torch
import torch.nn as nn
import fm

class ProtRNA(nn.Module):
    def __init__(self):
        super().__init__()

        self.rna_model, self.rna_alphabet = fm.pretrained.rna_fm_t12()

        self.dense1 = nn.Linear(640, 64)
        self.dense2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)
        self.act = nn.Sigmoid()
        self.elu = nn.ELU()

        for m in self.children():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)

    def get_alphabet(self):
        return self.rna_alphabet

    def forward(self, tokens):
        rna_results = self.rna_model(tokens, repr_layers=[12])
        seq_emb = rna_results["representations"][12]
        seq_emb = seq_emb[:,0,:]

        dense1 = self.elu(self.dropout(self.dense1(seq_emb)))
        dense2 = self.dense2(self.dropout(dense1))
        out = self.act(dense2)

        return out