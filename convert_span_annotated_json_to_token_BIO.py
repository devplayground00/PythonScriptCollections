import json
import re
import csv

def tokenize(text):
    # Tokenize words and keep punctuation
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def map_tokens_to_char_spans(text, tokens):
    spans = []
    current_pos = 0
    for token in tokens:
        start = text.find(token, current_pos)
        if start == -1:
            raise ValueError(f"Token '{token}' not found starting from {current_pos}")
        end = start + len(token)
        spans.append((start, end))
        current_pos = end
    return spans

def assign_bio_labels(tokens, spans, entities):
    labels = ["O"] * len(tokens)
    for ent in entities:
        start, end = ent["start"], ent["end"]
        ent_label = ent["labels"][0]
        inside = False
        for i, (tok_start, tok_end) in enumerate(spans):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            if not inside:
                labels[i] = f"B-{ent_label}"
                inside = True
            else:
                labels[i] = f"I-{ent_label}"
    return labels

def process_json_file(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)

        writer.writerow(["Token", "Label"])

        for entry in data:
                text = entry["data"]["text"]
                annotations = entry.get("annotations", [])
                if not annotations or not annotations[0].get("result"):
                    continue
                entities = [res["value"] for res in annotations[0]["result"]]

                tokens = tokenize(text)
                spans = map_tokens_to_char_spans(text, tokens)
                labels = assign_bio_labels(tokens, spans, entities)

                for token, label in zip(tokens, labels):
                    writer.writerow([str(token), str(label)])
                writer.writerow([])

    print(f"CSV-style BIO NER data saved to: {output_path}")

# === Run it ===
if __name__ == "__main__":
    process_json_file(
        r"C:\Users\pc\Downloads\project-2-at-2025-05-31-16-33-6f78c105.json",
        r"C:\Users\pc\Desktop\ner_output.csv"
    )

