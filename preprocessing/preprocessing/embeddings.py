from functools import lru_cache
import numpy as np

from transformers import TFBertModel, BertTokenizer
import tensorflow as tf


@lru_cache(maxsize=1)
def get_embedding_model():
    # First time this is executed, the model is downloaded. Actually, the download process should be done
    # a side, from a controlled version of the model, included, for instance in the docker build process.
    model = TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer


def embed(texts):
    model, tokenizer = get_embedding_model()

    embeddings = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tf.constant(tokens)[None, :]
        outputs = model(tokens)
        embeddings.append(outputs[1][0])

    return np.array(embeddings)
