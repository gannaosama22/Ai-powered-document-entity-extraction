"""Run inference experiments with the project's Baseline NER model.

This script does not train or fine-tune a model. It uses en_core_web_sm by
default, or a trained spaCy model directory supplied with --model.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from baseline_ner import load_model, predict_entities


def run_experiments(
    input_path: str,
    output_dir: str = "results",
    text_column: str | None = None,
    model_name: str | None = None,
    limit: int | None = None,
) -> dict:
    data = pd.read_csv(input_path)
    if text_column is None:
        # Prefer source text because the preprocessing notebook warns that
        # cleaning can change character offsets.
        text_column = "original_text" if "original_text" in data.columns else "message"
    if text_column not in data.columns:
        raise ValueError(
            f"Text column {text_column!r} not found. Available columns: "
            f"{', '.join(map(str, data.columns))}"
        )

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        data = data.head(limit).copy()

    nlp = load_model(model_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    document_rows = []
    label_counts = Counter()
    total_entities = 0

    for row_number, value in enumerate(data[text_column].tolist()):
        text = value if isinstance(value, str) else ""
        entities = predict_entities(text, nlp)
        total_entities += len(entities)
        label_counts.update(entity["label"] for entity in entities)

        doc_id = row_number
        document_rows.append({
            "document_id": doc_id,
            "source_row": row_number,
            "text_length": len(text),
            "entity_count": len(entities),
        })
        for entity_number, entity in enumerate(entities, start=1):
            prediction_rows.append({
                "document_id": doc_id,
                "entity_number": entity_number,
                **entity,
            })

    predictions = pd.DataFrame(
        prediction_rows,
        columns=["document_id", "entity_number", "text", "label",
                 "start_char", "end_char"],
    )
    documents = pd.DataFrame(
        document_rows,
        columns=["document_id", "source_row", "text_length", "entity_count"],
    )
    results = pd.DataFrame(
        [{"label": label, "entity_count": count}
         for label, count in sorted(label_counts.items())],
        columns=["label", "entity_count"],
    )

    predictions.to_csv(output_path / "predictions.csv", index=False, encoding="utf-8-sig")
    documents.to_csv(output_path / "document_summary.csv", index=False, encoding="utf-8-sig")
    results.to_csv(output_path / "experiment_results.csv", index=False, encoding="utf-8-sig")

    report = {
        "model": model_name or "en_core_web_sm",
        "input_file": str(input_path),
        "text_column": text_column,
        "documents_processed": len(data),
        "documents_with_entities": int((documents["entity_count"] > 0).sum()),
        "total_entities_predicted": total_entities,
        "entity_counts_by_label": dict(sorted(label_counts.items())),
        "evaluation_metrics": None,
        "evaluation_note": (
            "Precision, recall, and F1 were not calculated because this project "
            "input has no gold-standard entity annotations."
        ),
    }
    (output_path / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cleaned_dataset.csv")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--model", default=None,
                        help="Optional path to a trained spaCy model; defaults to en_core_web_sm")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of first rows to process")
    args = parser.parse_args()
    report = run_experiments(
        input_path=args.input,
        output_dir=args.output_dir,
        text_column=args.text_column,
        model_name=args.model,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
