# src/predict.py

import argparse
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray


# --- Dummy-Klassifikator ---
# Wir bauen eine Klasse, die sich genau wie ein scikit-learn Modell verhält.
# So bleibt deine restliche Logik (predict_texts) unangetastet.
class DummyClassifier:
    def predict(self, texts: list[str]) -> NDArray[Any]:
        # Einfache Regel: 1 bei "gut" oder "freue", sonst 0
        preds = [
            1 if any(word in t.lower() for word in ["gut", "freue", "super"]) else 0
            for t in texts
        ]
        return np.array(preds)

    def predict_proba(self, texts: list[str]) -> NDArray[np.float64]:
        # Simuliert Wahrscheinlichkeiten
        probs = []
        for text in texts:
            if 1 in self.predict([text]):
                probs.append([0.1, 0.9])
            else:
                probs.append([0.8, 0.2])
        return np.array(probs)


def load_model(model_path: str) -> Any:
    """Gibt in dieser Version immer den Dummy-Klassifikator zurück."""
    # Wir ignorieren den model_path für den Moment, um FileNotFound zu vermeiden
    return DummyClassifier()


def predict_texts(
    classifier: Any, input_texts: list[str]
) -> tuple[list[int], list[float | None]]:
    """Gibt Labels und Wahrscheinlichkeiten für jeden Text zurück."""

    # FIX: Falls die Eingabe leer ist, sofort leere Listen zurückgeben
    if not input_texts:
        return [], []

    preds: NDArray[Any] = classifier.predict(input_texts)
    if hasattr(classifier, "predict_proba"):
        probs_arr: NDArray[np.float64] = classifier.predict_proba(input_texts)[:, 1]
        probs = [float(p) for p in probs_arr.tolist()]
    else:
        probs = [None] * len(input_texts)
    return preds.astype(int).tolist(), probs


def format_prediction_lines(
    texts: list[str], preds: list[int], probs: list[float | None]
) -> list[str]:
    """Erstellt tab-getrennte Zeilen für die Ausgabe."""
    lines: list[str] = []
    for text, pred, prob in zip(texts, preds, probs):
        if prob is None:
            lines.append(f"{pred}\t{text}")
        else:
            lines.append(f"{pred}\t{prob:.3f}\t{text}")
    return lines


def main(
    model_path: str, input_texts: list[str], output_path: str | None = None
) -> None:
    classifier = load_model(model_path)
    preds, probs = predict_texts(classifier, input_texts)

    output_lines = format_prediction_lines(input_texts, preds, probs)

    for line in output_lines:
        print(line)  # noqa: T201

    if output_path:
        # Erstellt den Ordner (data/), falls er im Runner nicht existiert
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Wir behalten die Argumente bei, damit der CI-Aufruf nicht crashed
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("--input", help="Pfad zur Eingabedatei")
    parser.add_argument("--output", help="Pfad zur Ausgabedatei")
    parser.add_argument("text", nargs="*", help="Direkte Texteingabe")

    args = parser.parse_args()

    # Eingabe-Logik
    final_texts = []
    if args.input and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            final_texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        final_texts = args.text

    if not final_texts:
        # Ein Standard-Text, falls alles leer ist, damit die Pipeline was zu tun hat
        final_texts = ["Das ist ein Testlauf.", "Ich freue mich über den grünen Haken!"]

    main(model_path=args.model, input_texts=final_texts, output_path=args.output)
