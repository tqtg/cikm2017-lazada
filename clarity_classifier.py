import csv
import os
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return True
    return False

def extract_features(filename):
    features = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            title = row[2]
            features.append([len(title), contains_number(title)])
    return np.asarray(features)

def write_submission(model):
    if not os.path.exists('submission'):
        os.makedirs('submission')
    X = extract_features("data/validation/data_valid.csv")
    y = model.predict_proba(X)[:,1]
    np.savetxt('submission/clarity_valid.predict', y, fmt='%.5f')
    print('clarity_valid.predict updated')

if __name__ == "__main__":
    X = extract_features("data/training/data_train.csv")
    y = np.loadtxt("data/training/clarity_train.labels", dtype=int)

    model = LogisticRegression()
    model.fit(X, y)
    print("Model RMSE: %f" % mean_squared_error(model.predict_proba(X)[:,1], y)**0.5)

    write_submission(model)