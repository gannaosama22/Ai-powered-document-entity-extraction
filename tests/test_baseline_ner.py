import json
import tempfile
import unittest
from pathlib import Path

import spacy

from baseline_ner.model import predict_entities
from baseline_ner.train import AnnotationFormatError, load_annotations, train_model


class BaselineNerTests(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")

    def test_loads_valid_jsonl_annotations(self):
        text = "Email Alice Smith about Enron."
        record = {
            "text": text,
            "entities": [
                {"start": 6, "end": 17, "label": "PERSON"},
                {"start": 24, "end": 29, "label": "ORG"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            examples = load_annotations(path, self.nlp)

        self.assertEqual(
            [(entity.text, entity.label_) for entity in examples[0].reference.ents],
            [("Alice Smith", "PERSON"), ("Enron", "ORG")],
        )

    def test_rejects_missing_entity_label(self):
        record = {"text": "Alice Smith", "entities": [{"start": 0, "end": 11}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(AnnotationFormatError, "missing a non-empty label"):
                load_annotations(path, self.nlp)

    def test_rejects_overlapping_spans(self):
        record = {
            "text": "Alice Smith",
            "entities": [
                {"start": 0, "end": 11, "label": "PERSON"},
                {"start": 0, "end": 5, "label": "PERSON"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(AnnotationFormatError, "must not overlap"):
                load_annotations(path, self.nlp)

    def test_training_saves_a_loadable_model(self):
        records = [
            {"text": "Enron hired Alice.", "entities": [
                {"start": 0, "end": 5, "label": "ORG"},
                {"start": 12, "end": 17, "label": "PERSON"},
            ]},
            {"text": "Alice works at Enron.", "entities": [
                {"start": 0, "end": 5, "label": "PERSON"},
                {"start": 15, "end": 20, "label": "ORG"},
            ]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "train.jsonl"
            model_path = Path(directory) / "model"
            data_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            scores = train_model(data_path, model_path, epochs=1, dropout=0)

            self.assertIn("ents_f", scores)
            trained_nlp = spacy.load(model_path)
            self.assertEqual(trained_nlp.get_pipe("ner").labels, ("ORG", "PERSON"))

    def test_prediction_includes_original_character_offsets(self):
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        ruler.add_patterns([{"label": "ORG", "pattern": "Enron"}])

        entities = predict_entities("Email Enron today.", nlp)

        self.assertEqual(
            entities,
            [{"text": "Enron", "label": "ORG", "start_char": 6, "end_char": 11}],
        )


if __name__ == "__main__":
    unittest.main()
