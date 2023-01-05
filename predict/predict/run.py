import json
import argparse
import os
import time
from collections import OrderedDict

from tensorflow.keras.models import load_model
from numpy import argsort

from preprocessing.preprocessing.embeddings import embed

import logging
logger = logging.getLogger(__name__)
import tensorflow as tf

class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    def __eq__(self, target):
        return\
            self.model.__dir__() == target.model.__dir__() and\
            self.params == target.params and\
            self.labels_to_index == target.labels_to_index and\
            self.labels_index_inv == target.labels_index_inv

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # load model
        model = load_model(artefacts_path)

        # load params
        params = json.load(open(artefacts_path+'/params.json'))

        # load labels_to_index
        labels_to_index = json.load(open(artefacts_path+'/labels_index.json'))

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=5):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()

        logger.info(f"Predicting text_list=`{text_list}`")

        # embed text_list
        embedding = embed(text_list)

        # predict tags indexes from embeddings
        predict = self.model.predict(embedding)

        # from tags indexes compute top_k tags for each text
        indices = argsort(predict)[-top_k:]
        list_indices = [index.argmin() for index in indices]
        predictions = [self.labels_index_inv[index] for index in list_indices]
        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))
