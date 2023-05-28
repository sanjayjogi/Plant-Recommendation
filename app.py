from flask import Flask, request, Markup, abort
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger
import warnings

# from flask_cors import CORS

app = Flask(__name__)
Swagger(app)
# CORS(app)

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


@app.route('/fertilizer', methods=['GET'])
def fert_recommend():
    """Fertilizer Recommendation
    ---
    parameters:
        # - name: model
        #   in: query
        #   type: string
        #   require: true
        - name: cropname
          in: query
          type: string
          require: true
        - name: nitrogen
          in: query
          type: number
          require: true
        - name: phosphorous
          in: query
          type: number
          require: true
        - name: pottasium
          in: query
          type: number
          require: true
    responses:
          200:
              description: the output values
    """

    crop_name = request.args.get('cropname')
    N = int(request.args.get('nitrogen'))
    P = int(request.args.get('phosphorous'))
    K = int(request.args.get('pottasium'))
    # N = (request.args.get('nitrogen'))
    # P = (request.args.get('phosphorous'))
    # K = (request.args.get('pottasium'))
    # ph = int(request.args.get('ph'))

    df = pd.read_csv('fertilizer.csv')
    if not df['Crop'].isin([crop_name]).any():
        print('The value doesnt exists in the column')

        abort(400)

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
            response = {
                "statement": "The N value of soil is high and might give rise to weeds",
                "recommendation": [{
                    "1":  "Manure  – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.",
                    "2": "Coffee grinds  – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.",
                    "3": "Plant nitrogen fixing plants – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil",
                    "4": "Plant ‘green manure’ crops like cabbage, corn and brocolli",
                    "5": "Use mulch (wet grass) while growing crops - Mulch can also include sawdust and scrap soft woods"
                }]
            }
        else:
            key = "Nlow"
            response = {
                "statement": "The N value of your soil is low.",
                "recommendation": [{
                    "1":  "Add sawdust or fine woodchips to your soil – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.",
                    "2": "Plant heavy nitrogen feeding plants – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.",
                    "3": "Water – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.",
                    "4": "Sugar – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.",
                    "5": "Add composted manure to the soil.",
                    "6": "Plant Nitrogen fixing plants like peas or beans.",
                    "7": "Use NPK fertilizers with high N value.",
                    "8": "Do nothing – It may seem counter-intuitive, but if you already have plants that are producing lots of foliage, it may be best to let them continue to absorb all the nitrogen to amend the soil for your next crops."
                }]
            }
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
            response = {
                "statement": "The P value of your soil is high.",
                "recommendation": [{
                    "1":  "Avoid adding manure – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.",
                    "2": "Use only phosphorus-free fertilizer – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.",
                    "3": "Water your soil – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.",
                    "4": "Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).",
                    "5": "Use crop rotations to decrease high phosphorous levels"
                }]
            }
        else:
            key = "Plow"
            response = {
                "statement": "The N value of your soil is low.",
                "recommendation": [{
                    "1":  "Bone meal – a fast acting source that is made from ground animal bones which is rich in phosphorous.",
                    "2": "Rock phosphate – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.",
                    "3": "Phosphorus Fertilizers – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).",
                    "4": "Organic compost – adding quality organic compost to your soil will help increase phosphorous content.",
                    "5": "Manure – as with compost, manure can be an excellent source of phosphorous for your plants.",
                    "6": "Clay soil – introducing clay particles into your soil can help retain & fix phosphorus deficiencies.",
                    "7": "Ensure proper soil pH – having a pH in the 6.0 to 7.0 range has been scientifically proven to have the optimal phosphorus uptake in plants.",
                    "8": "If soil pH is low, add lime or potassium carbonate to the soil as fertilizers. Pure calcium carbonate is very effective in increasing the pH value of the soil.",
                    "9": "If pH is high, addition of appreciable amount of organic matter will help acidify the soil. Application of acidifying fertilizers, such as ammonium sulfate, can help lower soil pH."
                }]
            }
    else:
        if k < 0:
            key = 'KHigh'
            response = {
                "statement": "The K value of your soil is high",
                "recommendation": [{
                    "1":  "Loosen the soil deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.",
                    "2": "Sift through the soil, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.",
                    "3": "Phosphorus Fertilizers – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).",
                    "4": "Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.",
                    "5": "Use NPK fertilizers with low K levels and organic fertilizers since they have low NPK values.",
                    "6": "Grow a cover crop of legumes that will fix nitrogen in the soil. This practice will meet the soil’s needs for nitrogen without increasing phosphorus or potassium."
                }]
            }
        else:
            key = "Klow"
            response = {
                "statement": "The K value of your soil is low.",
                "recommendation": [{
                    "1":  "Mix in muricate of potash or sulphate of potash",
                    "2": " Try kelp meal or seaweed",
                    "3": "Try Sul-Po-Mag",
                    "4": "Bury banana peels an inch below the soils surface",
                    "5": " Use Potash fertilizers since they contain high values potassium"
                }]
            }

    return response


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0')
