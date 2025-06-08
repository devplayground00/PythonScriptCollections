import spacy

# Load the best model
nlp = spacy.load(r"C:\Users\pc\Downloads\distilrebertabase\model-best")

# Test on a sample
doc = nlp("")

for ent in doc.ents:
    print(ent.text, ent.label_)
