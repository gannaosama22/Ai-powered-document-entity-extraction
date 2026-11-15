# Baseline NER handoff — Ahmed Hamdy + Ahmed Ali

## Ahmed Hamdy — Baseline NER implementation/training
- `baseline_ner/model.py`: loads `en_core_web_sm` or a saved spaCy model and extracts entities with source character offsets.
- `baseline_ner/train.py`: trains a custom English NER model from annotated JSONL and saves the best validation checkpoint.
- `tests/test_baseline_ner.py`: implementation and training checks.

**Status in the supplied ZIP:** implementation and training code are present. A custom-trained checkpoint is not present. The preprocessing notebook's Enron sample is not annotated, so it cannot be used as supervised NER training data as-is. This means the code handoff exists, but a completed custom training run cannot be verified from this archive.

## Ahmed Ali — Baseline experiments, predictions, results
- Added `baseline_ner/experiments.py`.
- Runs the baseline model over the CSV produced by the preprocessing notebook.
- Writes predictions, per-document summary, label counts, and a run report under `results/`.
- Uses original email text when available so character offsets remain tied to the source text.

## Run
1. Run the preprocessing notebook so it creates `cleaned_dataset.csv`.
2. Install requirements and download the spaCy baseline:
   `python -m pip install -r requirements.txt`
   `python -m spacy download en_core_web_sm`
3. Run:
   `python -m baseline_ner.experiments --input cleaned_dataset.csv --output-dir results`

## Limits / what is still needed
- No Enron gold entity annotations are included.
- No custom trained checkpoint is included.
- Precision, recall, F1, and a meaningful supervised evaluation cannot be reported until annotated ground truth is supplied.
- The default predictions are from spaCy's pretrained general English model, not an Enron-finetuned model.
