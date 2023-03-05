from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

dt_in = open('DecisionTree.pkl', 'rb')
DecisionTree = pickle.load(dt_in)

svm_in = open('SVM.pkl', 'rb')
SVM = pickle.load(svm_in)

rf_in = open('RF.pkl', 'rb')
RF = pickle.load(rf_in)


@app.route('/')
def welcome():
    return "acha"


@app.route('/predict', methods=['GET'])
def predic_note_authenticattion():
    """Authenticating Bank Notes
    ---
    parameters:
        - name: model
          in: query
          type: string
          require: true
        - name: n
          in: query
          type: number
          require: true
        - name: p
          in: query
          type: number
          require: true
        - name: k
          in: query
          type: number
          require: true
        - name: temperature
          in: query
          type: number
          require: true
        - name: humidity
          in: query
          type: number
          require: true
        - name: ph
          in: query
          type: number
          require: true
        - name: rainfall
          in: query
          type: number
          require: true
    responses:
          200:
              description: the output values
    """

    model = request.args.get('model')
    n = request.args.get('n')
    p = request.args.get('p')
    k = request.args.get('k')
    temperature = request.args.get('temperature')
    humidity = request.args.get('humidity')
    ph = request.args.get('ph')
    rainfall = request.args.get('rainfall')

    sample = [n, p, k, temperature, humidity, ph, rainfall]
    single_sample = np.array(sample).reshape(1, -1)

    if model == 'dt':
        prediction = DecisionTree.predict(single_sample)
    elif model == 'svm':
        prediction = SVM.predict(single_sample)
    elif model == 'rf':
        prediction = RF.predict(single_sample)
    else:
        prediction = ['No models found']

    pred = {
        'crop': prediction[0]
    }
    return pred


app.run(host='0.0.0.0', port=5000)
