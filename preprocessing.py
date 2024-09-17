# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(input_data, scaler):
    df = pd.DataFrame(input_data)
    return scaler.transform(df)

def fit_scaler(train_X):
    scaler = StandardScaler()
    scaler.fit(train_X)
    return scaler
