# train_layoutlmv3.py
import os
import json
from datasets import load_dataset, DatasetDict
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
from PIL import Image

# ------------------- Paths -------------------
BASE_DIR = "layoutlmv3_training"
DATA_DIR = r"C:\Users\ruqai\Desktop\data"
OUTPUT_DIR = r"C:\Users\ruqai\Desktop\layoutlmv3_output"
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# ------------------- Load Dataset -------------------
# Must contain: "id", "words", "bboxes", "ner_tags", "image"
def load_custom_dataset(data_dir):
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "train.json"),
        "validation": os.path.join(data_dir, "valid.json")
    })
    return dataset

dataset = load_custom_dataset(DATA_DIR)

# ------------------- Labels -------------------
unique_labels = set()
for example in dataset["train"]:
    unique_labels.update(example["ner_tags"])
unique_labels = sorted(set(unique_labels))

label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# ------------------- Processor -------------------
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def preprocess(example):
    words = example["words"]
    boxes = example["bboxes"]
    image = Image.open(example["image"]).convert("RGB")
    word_labels = [label2id[label] for label in example["ner_tags"]]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=word_labels,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # <--- return tensor directly
    )

    # Remove batch dim (shape [1, ...] -> [...]) for Trainer
    return {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.size(0) == 1 else v for k, v in encoding.items()}


# Apply preprocessing
encoded_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# ------------------- Model -------------------
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# ------------------- Training -------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=15,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer=processor.tokenizer, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=processor.tokenizer,
    data_collator=data_collator
)

# ------------------- Train -------------------
trainer.train()

# ------------------- Save -------------------
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
