class Stats:
    def __init__(self, conf_mx, ba, f1, g, i, fpr, tpr):
        super().__init__()
        self.conf_matrix = conf_mx
        self.ba = ba
        self.f1 = f1
        self.g = g
        self.i = i
        self.fpr = fpr
        self.tpr = tpr


from save_data import save_data
class NetworkParams:
    def __init__(self, name, desc, filename, hyperparameters, resulting_hyperparameters, stats, auc_train, auc_test, comment):
        super().__init__()
        self.name = name
        self.desc = desc
        self.filename = filename
        self.hyperparameters = hyperparameters
        self.resulting_hyperparameters = resulting_hyperparameters
        self.stats = stats
        self.auc = (auc_train, auc_test)
        self.comment = comment

    def save(self):
        save_data(self.name, self.desc, self.filename, self.hyperparameters, self.resulting_hyperparameters, self.stats, self.auc, self.comment)