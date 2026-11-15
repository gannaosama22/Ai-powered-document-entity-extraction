# AI-Powered Document Entity Extraction

This project currently contains an Enron email preprocessing notebook. It
downloads a 1,000-row sample and performs cleaning, NLTK tokenization, and
spaCy POS tagging. Those emails have no entity annotations; the notebook does
not train or evaluate an NER model. Its cleaner also changes text, so use the
original email text when character offsets need to refer to source text.

## Baseline NER

`baseline_ner` provides two distinct paths:

- **Pretrained inference:** spaCy's `en_core_web_sm` provides an immediately
  usable English NER baseline. It has not been fine-tuned on Enron email.
- **Supervised training:** `train.py` builds a new English NER model from
  explicitly annotated JSONL. It does not create labels from raw emails.

The pretrained model recognizes its own standard entity labels, including
`PERSON`, `ORG`, `GPE`, `DATE`, `TIME`, `MONEY`, and `CARDINAL`. The project has
not specified a custom label schema; training labels are taken from the
annotation file.

Install dependencies and the pretrained baseline:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Use the pretrained model or load a trained model directory:

```python
from baseline_ner import load_model, predict_entities

nlp = load_model()  # pretrained en_core_web_sm; inference only
entities = predict_entities("Enron contacted Alice Smith in Houston.", nlp)
print(entities)
```

Each result contains `text`, `label`, `start_char`, and `end_char`; offsets
refer to the exact input string. To load a model produced by the training
pipeline, pass its directory to `load_model("models/custom_ner")`.

## Annotated training data

Create a UTF-8 JSONL file with one object per email. Character offsets use
Python's zero-based, end-exclusive convention and must align with spaCy token
boundaries. Use an empty entity list only when the text has been reviewed and
contains no entities:

```json
{"text":"Email Alice Smith about Enron.","entities":[{"start":6,"end":17,"label":"PERSON"},{"start":24,"end":29,"label":"ORG"}]}
```

Missing labels, invalid offsets, overlapping spans, and token-misaligned spans
are rejected with the source line number. Supply separate validation data, or
the trainer will reserve a deterministic 20% split:

```powershell
python -m baseline_ner.train --train data/train.jsonl --validation data/validation.jsonl --output models/custom_ner --epochs 20
```

The trainer reports validation scores and saves the best checkpoint, including
spaCy's model configuration, to the output directory. With no annotations,
training cannot be run and no trained project model is produced.

Run the focused checks from this folder:

```powershell
python -m unittest discover -s tests -v
```
