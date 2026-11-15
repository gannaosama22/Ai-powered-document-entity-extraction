"""Train a spaCy NER model from project-annotated JSONL data."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.training import Example
from spacy.util import compounding, minibatch


class AnnotationFormatError(ValueError):
    """Raised when an annotation file is malformed or cannot align to tokens."""


def load_annotations(path: str | Path, nlp: Language) -> list[Example]:
    """Read JSONL rows with text and character-based entity annotations."""
    examples = []
    with Path(path).open(encoding="utf-8") as annotation_file:
        for line_number, line in enumerate(annotation_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                examples.append(_make_example(record, nlp))
            except (json.JSONDecodeError, AnnotationFormatError) as error:
                raise AnnotationFormatError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise AnnotationFormatError(f"{path}: no annotated examples were found")
    return examples


def _make_example(record: Any, nlp: Language) -> Example:
    if not isinstance(record, dict):
        raise AnnotationFormatError("each JSONL row must be an object")

    text = record.get("text")
    entities = record.get("entities")
    if not isinstance(text, str) or not text.strip():
        raise AnnotationFormatError("'text' must be a non-empty string")
    if not isinstance(entities, list):
        raise AnnotationFormatError("'entities' must be a list (use [] for no entities)")

    spans = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise AnnotationFormatError(f"entity {index} must be an object")
        start = entity.get("start")
        end = entity.get("end")
        label = entity.get("label")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)):
            raise AnnotationFormatError(f"entity {index} start and end must be integers")
        if not isinstance(label, str) or not label.strip():
            raise AnnotationFormatError(f"entity {index} is missing a non-empty label")
        if start < 0 or end <= start or end > len(text):
            raise AnnotationFormatError(f"entity {index} has an invalid character span")
        spans.append((start, end, label))

    spans.sort(key=lambda span: (span[0], span[1]))
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise AnnotationFormatError("entity spans must not overlap")

    doc = nlp.make_doc(text)
    for start, end, label in spans:
        if doc.char_span(start, end, label=label, alignment_mode="strict") is None:
            raise AnnotationFormatError(
                f"entity span ({start}, {end}) is not aligned to spaCy token boundaries"
            )

    return Example.from_dict(doc, {"entities": spans})


def _split_examples(
    examples: list[Example], validation_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    if len(examples) < 2:
        raise ValueError(
            "At least two annotated examples are required when validation data "
            "is not supplied."
        )
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def train_model(
    train_path: str | Path,
    output_path: str | Path,
    validation_path: str | Path | None = None,
    epochs: int = 20,
    dropout: float = 0.2,
    seed: int = 13,
) -> dict[str, float]:
    """Train a new English NER model and save its best validation checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1)")

    nlp = spacy.blank("en")
    train_examples = load_annotations(train_path, nlp)
    if validation_path is None:
        train_examples, validation_examples = _split_examples(
            train_examples, validation_fraction=0.2, seed=seed
        )
    else:
        validation_examples = load_annotations(validation_path, nlp)

    labels = {
        entity.label_
        for example in train_examples
        for entity in example.reference.ents
    }
    if not labels:
        raise ValueError("Training data must contain at least one labeled entity.")
    validation_labels = {
        entity.label_
        for example in validation_examples
        for entity in example.reference.ents
    }
    unknown_validation_labels = validation_labels - labels
    if unknown_validation_labels:
        raise ValueError(
            "Validation data contains labels absent from training data: "
            + ", ".join(sorted(unknown_validation_labels))
        )

    ner = nlp.add_pipe("ner")
    for label in sorted(labels):
        ner.add_label(label)

    nlp.initialize(get_examples=lambda: train_examples)
    random_generator = random.Random(seed)
    best_f_score = -1.0
    best_scores: dict[str, float] = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(epochs):
        random_generator.shuffle(train_examples)
        losses: dict[str, float] = {}
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=dropout, losses=losses)

        scores = nlp.evaluate(validation_examples)
        f_score = float(scores.get("ents_f", 0.0))
        if f_score > best_f_score:
            best_f_score = f_score
            best_scores = {
                key: float(value)
                for key, value in scores.items()
                if isinstance(value, (int, float))
            }
            nlp.to_disk(output_path)

    return best_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Annotated training JSONL file")
    parser.add_argument("--validation", help="Optional annotated validation JSONL file")
    parser.add_argument(
        "--output", default="models/custom_ner", help="Directory to save the model"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    try:
        scores = train_model(
            train_path=args.train,
            validation_path=args.validation,
            output_path=args.output,
            epochs=args.epochs,
            dropout=args.dropout,
            seed=args.seed,
        )
    except (AnnotationFormatError, OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Training completed. Best validation scores: {json.dumps(scores)}")
    print(f"Saved trained model to: {Path(args.output)}")


if __name__ == "__main__":
    main()
