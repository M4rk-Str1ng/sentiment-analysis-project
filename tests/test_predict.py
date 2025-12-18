# tests/test_predict.py

import unittest
import numpy as np
# Wir importieren die Logik direkt aus deinem src-Ordner
from src.predict import DummyClassifier, predict_texts

class TestPredictLogic(unittest.TestCase):
    """
    Diese Klasse testet, ob unsere Vorhersage-Logik korrekt arbeitet.
    In der CI/CD Pipeline wird dieser Test automatisch ausgeführt.
    """

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt."""
        cls.classifier = DummyClassifier()

    def test_positive_sentiment(self):
        """Prüft, ob 'gut' als positives Sentiment (1) erkannt wird."""
        texts = ["Das ist ein gutes Ergebnis"]
        preds, _ = predict_texts(self.classifier, texts)
        self.assertEqual(preds[0], 1, "Positives Wort wurde nicht als 1 erkannt.")

    def test_negative_sentiment(self):
        """Prüft, ob neutrale/andere Texte als 0 erkannt werden."""
        texts = ["Das Wetter ist heute normal."]
        preds, _ = predict_texts(self.classifier, texts)
        self.assertEqual(preds[0], 0, "Neutraler Text sollte Label 0 erhalten.")

    def test_empty_input(self):
        """Sicherheits-Check: Was passiert bei leeren Listen?"""
        preds, probs = predict_texts(self.classifier, [])
        self.assertEqual(len(preds), 0)
        self.assertEqual(len(probs), 0)

if __name__ == "__main__":
    unittest.main() 

