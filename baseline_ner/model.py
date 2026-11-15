from pathlib import Path

import spacy
from spacy.language import Language


DEFAULT_MODEL = "en_core_web_sm"


def load_model(model_path: str | Path | None = None) -> Language:
    """Load a trained project model or spaCy's pretrained English baseline."""
    model_name = str(model_path) if model_path is not None else DEFAULT_MODEL
    try:
        return spacy.load(model_name)
    except OSError as error:
        if model_path is None:
            raise OSError(
                f"Could not load {DEFAULT_MODEL!r}. Install it with "
                "`python -m spacy download en_core_web_sm`, or pass the path "
                "to a trained project model."
            ) from error
        raise OSError(f"Could not load spaCy model from {model_name!r}.") from error


def predict_entities(text: str, nlp: Language) -> list[dict[str, str | int]]:
    """Return entity spans and character offsets for the supplied text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    doc = nlp(text)
    return [
        {
            "text": entity.text,
            "label": entity.label_,
            "start_char": entity.start_char,
            "end_char": entity.end_char,
        }
        for entity in doc.ents
    ]
