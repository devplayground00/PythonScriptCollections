import spacy
from spacy.tokens import DocBin
import random
import hashlib
from pathlib import Path

# File paths
input_path = Path(r"C:\Users\pc\Desktop\NER 062025\ner_data.spacy")
train_output_path = Path(r"C:\Users\pc\Desktop\NER 062025\ner_data_train.spacy")
dev_output_path = Path(r"C:\Users\pc\Desktop\NER 062025\ner_data_dev.spacy")

# Settings
eval_split = 0.2
random.seed(42)

# Load data
nlp = spacy.blank("en")
doc_bin = DocBin().from_disk(str(input_path))
docs = list(doc_bin.get_docs(nlp.vocab))

# Remove duplicate docs by byte hash
seen = {}
for doc in docs:
    key = doc.text.strip()  # deduplicate based on cleaned raw text
    if key not in seen:
        seen[key] = doc

unique_docs = list(seen.values())
random.shuffle(unique_docs)

# Split cleanly
split = int(len(unique_docs) * (1 - eval_split))
train_docs = unique_docs[:split]
dev_docs = unique_docs[split:]

# Verify: Check for overlap
train_hashes = {hashlib.md5(doc.to_bytes()).hexdigest() for doc in train_docs}
dev_hashes = {hashlib.md5(doc.to_bytes()).hexdigest() for doc in dev_docs}
overlap = train_hashes.intersection(dev_hashes)

# Save to disk
DocBin(docs=train_docs).to_disk(str(train_output_path))
DocBin(docs=dev_docs).to_disk(str(dev_output_path))

# Report
print(f"Split complete.")
print(f"Train docs: {len(train_docs)}")
print(f"Dev docs: {len(dev_docs)}")
print(f"Unique docs total: {len(unique_docs)}")
print(f"Overlapping hashes between train/dev: {len(overlap)}")

if overlap:
    print("WARNING: Duplicate documents still found between train and dev!")
else:
    print("No overlapping documents.")
