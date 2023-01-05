#!/usr/bin/env python3
import sys
sys.path.insert(0, sys.path[0]+'/../../')

from flask import Flask, render_template, request
import json
from run import TextPredictionModel

app = Flask(__name__)

@app.route('/')
def home():
   return 'Welcome to my wonderful micro-service! 🙂 Go to /predict to predict. Thx 🙂 (ilu Ismail)'

@app.route('/predict', methods=['POST'])
def predict():
    body=json.loads(request.get_data())
    text_list = body['textsToPredict']
    top_k = body['top_k']
    textPredictionModel = TextPredictionModel.from_artefacts('./path')
    label_list = textPredictionModel.predict(text_list, top_k=top_k)
    print(label_list)
    return str(label_list)

app.run(use_reloader = True, debug = True)
