
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python file.py <modelname>")
        sys.exit(1)


    names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK', 'ALL']
    name = sys.argv[1]

    if name in names :
        for i in range(0, 100):
            model = tf.keras.models.load_model(f"./model/{name}Model{i}.tf")

            Data_dir = './data/%s/' % name
            test = np.load(Data_dir + '%s_test.npz' % name)
            X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']
            print("****************Testing %s cell line ****************" % (name))
            y_pred = model.predict([X_en_tes, X_pr_tes])
            auc = roc_auc_score(y_tes, y_pred)
            aupr = average_precision_score(y_tes, y_pred)
            print(f"model{i} acc")
            print("AUC : ", auc)
            print("AUPR : ", aupr)

    else:
        print(f"invalid cell line: {name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
