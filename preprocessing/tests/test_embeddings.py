import unittest

from preprocessing.preprocessing.embeddings import embed


class EmbeddingsTest(unittest.TestCase):
    def test_embed(self):
        embeddings = embed(['hello world'])
        self.assertEqual(embeddings.shape, (1, 768))
