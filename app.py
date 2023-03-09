from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger
import warnings

app = Flask(__name__)
Swagger(app)

dt_in = open('DecisionTree.pkl', 'rb')
DecisionTree = pickle.load(dt_in)

svm_in = open('SVM.pkl', 'rb')
SVM = pickle.load(svm_in)

rf_in = open('RF.pkl', 'rb')
RF = pickle.load(rf_in)

model_in = open('model.pkl', 'rb')
model = pickle.load(model_in)


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model


@app.route('/')
def welcome():
    return "acha"


@app.route('/predict', methods=['GET'])
def get_plant():
    """Plant Recommendation
    ---
    parameters:
        # - name: model
        #   in: query
        #   type: string
        #   require: true
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

    # model = request.args.get('model')
    n = request.args.get('n')
    p = request.args.get('p')
    k = request.args.get('k')
    temperature = request.args.get('temperature')
    humidity = request.args.get('humidity')
    ph = request.args.get('ph')
    rainfall = request.args.get('rainfall')

    sample = [n, p, k, temperature, humidity, ph, rainfall]
    single_sample = np.array(sample).reshape(1, -1)

    # if model == 'dt':
    #     prediction = DecisionTree.predict(single_sample)
    # elif model == 'svm':
    #     prediction = SVM.predict(single_sample)
    # elif model == 'rf':
    #     prediction = RF.predict(single_sample)
    # elif model == 'model':
    #     loaded_model = load_model('model.pkl')
    #     prediction = loaded_model.predict(single_sample)
    # else:
    #     prediction = ['No models found']

    loaded_model = load_model('model.pkl')
    prediction = loaded_model.predict(single_sample)

    pred = {
        'crop': prediction[0]
    }
    return pred


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0')
