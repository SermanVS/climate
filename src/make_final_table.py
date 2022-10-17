import glob
import pandas as pd
from pathlib import Path  
from config_reader import Config

def make_final_table():
    cfg = Config()
    paths = glob.glob('../temp/' + cfg.mode + '/' + cfg.metric + '/*.csv')
    
    out_filepath = Path('../results/' + cfg.mode + '/' + cfg.metric + '.csv')  
    out_filepath.parent.mkdir(parents=True, exist_ok=True)  
    
    
    titles_to_read = ['Description', 'Filename', 'Hyperparameters', 'Resulting_hyperparameters', \
            'Conf_mx_train_row1', 'Conf_mx_train_row2', 'Conf_mx_test_row1', 'Conf_mx_test_row2', \
            'BA_train', 'BA_test', 'F1_train', 'F1_test', 'G_train', 'G_test', 'I_train', 'I_test',\
            'fpr_train', 'fpr_test', 'tpr_train', 'tpr_test', 'auc_train', 'auc_test', 'Comment']

    titles_to_write = ['Id', 'Description', 'Filename', 'Hyperparameters', 'Resulting_hyperparameters', \
            'Conf_mx_train_row1', 'Conf_mx_train_row2', 'Conf_mx_test_row1', 'Conf_mx_test_row2', \
            'BA_train', 'BA_test', 'F1_train', 'F1_test', 'G_train', 'G_test', 'I_train', 'I_test',\
            'fpr_train', 'fpr_test', 'tpr_train', 'tpr_test', 'auc_train', 'auc_test', 'Comment']

    data = pd.DataFrame(columns=titles_to_write)

    for index, path in enumerate(paths):
        file_data = pd.read_csv(path, sep=',', names=titles_to_read)
        file_data['Id'] = index + 1
        data = pd.concat([data, file_data])

    data.to_csv(out_filepath, index=False)     
    return data

