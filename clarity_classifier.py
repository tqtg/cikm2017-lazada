import csv
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from utils import write_submission

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
            ''' 
                Feel free to create your amazing features here
                ...
            '''
            features.append([len(title), contains_number(title)])
    return np.asarray(features)

if __name__ == "__main__":
    # Data loading
    X = extract_features("data/training/data_train.csv")
    y = np.loadtxt("data/training/clarity_train.labels", dtype=int)

    # Model training
    model = LogisticRegression()
    model.fit(X, y)
    print("Model RMSE: %f" % mean_squared_error(model.predict_proba(X)[:,1], y)**0.5)

    # Validation predicting
    X_valid = extract_features("data/validation/data_valid.csv")
    predicted_results = model.predict_proba(X_valid)[:, 1]
    write_submission('clarity_valid.predict', predicted_results)