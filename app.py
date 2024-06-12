from flask import Flask, request, jsonify
import logging
import json
import numpy as np
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = joblib.load("salary_prediction_model.pkl")

# Define a route for receiving predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json['data']
        
        # Convert data to numpy array
        # data = np.array(data)
   
        X_input = pd.DataFrame(data)
        print(X_input)
        
        # Make prediction
        prediction = model.predict(X_input)
        print(prediction)
        
        # Convert prediction to list and return
        return jsonify({'prediction': prediction.tolist()}), 200
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='localhost', port=5000)
