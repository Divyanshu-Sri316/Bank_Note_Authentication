from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open('classifier.pkl', "rb")
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')  # This loads the HTML file from templates

@app.route('/predict', methods=["GET"])
def predict_note_authentication():
    variance = request.args.get('variance', type=float)
    skewness = request.args.get('skewness', type=float)
    curtosis = request.args.get('curtosis', type=float)
    entropy = request.args.get('entropy', type=float)
    
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return f"The predicted value is {prediction[0]}"

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return f"The predicted values are {list(prediction)}"

if __name__ == '__main__':
    app.run(debug=True)
