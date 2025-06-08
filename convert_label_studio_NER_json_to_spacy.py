import json
from pathlib import Path
from spacy.tokens import DocBin
import spacy


def labelstudio_to_spacy(json_path, output_path):
    nlp = spacy.blank("en")  # use blank model to create docs
    db = DocBin()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for task in data:
        text = task["data"]["text"]
        entities = []
        for ann in task["annotations"]:
            for r in ann["result"]:
                start = r["value"]["start"]
                end = r["value"]["end"]
                label = r["value"]["labels"][0]

                # ============ NEW CODE START (strip whitespace from entity spans) ============
                original_span = text[start:end]
                stripped_span = original_span.strip()

                if not stripped_span:
                    continue  # skip empty or all-whitespace spans

                # Adjust start and end to trimmed span
                leading_ws = len(original_span) - len(original_span.lstrip())
                trailing_ws = len(original_span) - len(original_span.rstrip())
                start += leading_ws
                end -= trailing_ws
                # ============ NEW CODE END ============

                entities.append((start, end, label))

        print(f"Entities: {entities}")

        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                spans.append(span)
            else:
                print(f"[!] Skipped misaligned span: '{text[start:end]}' in text: {text[:50]}...")  # optional logging

        spans = spacy.util.filter_spans(spans)
        doc.ents = spans

        if not spans:
            print(f"[i] Added doc with NO entities: {text[:60]}")

        db.add(doc)

    db.to_disk(output_path)


# Run conversion
labelstudio_to_spacy(
    r"C:\Users\pc\Desktop\NER 062025\all_tasks_export.json",
    r"C:\Users\pc\Desktop\NER 062025\ner_data.spacy"
)
