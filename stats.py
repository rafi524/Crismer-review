class Stats:
    def __init__(self):
        self.acc = 0
        self.pre = 0
        self.re = 0
        self.f1 = 0
        self.roc = 0
        self.prc = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0
    def print(self):
        print('Accuracy: %.4f' %self.acc)
        print('Precision: %.4f' %self.pre)
        print('Recall: %.4f' %self.re)
        print('F1 Score: %.4f' %self.f1)
        print('ROC: %.4f' %self.roc)
        print('PR AUC: %.4f' %self.prc)
        print("Confusion Matrix")
        print(self.tn, "\t", self.fp)
        print(self.fn, "\t", self.tp)