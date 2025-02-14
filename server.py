from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('classifier.pkl', 'rb'))
label_encoder = pickle.load(open('fertilizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([data['features']]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        fertilizer_name = label_encoder.classes_[prediction[0]]

        return jsonify({'prediction': fertilizer_name})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
