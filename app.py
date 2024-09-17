from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import create_model, train_model
from preprocessing import fit_scaler
from predict import make_prediction

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('alloy-confp-train-data_v2.csv')
train_dataset = data.sample(frac=0.8, random_state=0)
Xcols = train_dataset.columns[train_dataset.columns.str.contains("C.")]
train_X = train_dataset[Xcols]
scaler = fit_scaler(train_X)

# Define model parameters
nn_params = {
    'act_func': 'relu',
    'nhidden_layer': 10,
    'bnorm': True,
    'l2_reg': None,
    'dropout': True,
}

# Create and train the model
model = create_model(nn_params, input_shape=(len(Xcols),))
train_y = train_dataset['HV']
val_dataset = data.drop(train_dataset.index)
val_X = val_dataset[Xcols]
val_y = val_dataset['HV']
history = train_model(model, train_X, train_y, val_X, val_y, epochs=300)

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.getlist('features')
    data = [[float(x) for x in data]]  # Convert to list of floats
    predictions = make_prediction(data, scaler)
    return jsonify({'prediction': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
