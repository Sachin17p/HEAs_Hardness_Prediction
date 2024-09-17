# predict.py
from model import load_model
from preprocessing import preprocess_input

def make_prediction(input_data, scaler):
    model = load_model()
    processed_data = preprocess_input(input_data, scaler)
    predictions = model.predict(processed_data)
    return predictions.tolist()
