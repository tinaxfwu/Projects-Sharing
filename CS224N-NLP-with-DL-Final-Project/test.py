import spacy

nlp = spacy.load("en_core_web_sm")
tag_to_ix = {"ADJ": 0, 
             "ADP": 1,
            "ADV": 2,
            "AUX": 3,
            "CCONJ": 4,
            "DET": 5,
            "INTJ": 6,
            "NOUN": 7,
            "NUM": 8,
            "PART": 9,
            "PRON": 10,
            "PROPN": 11,
            "PUNCT": 12,
            "SCONJ": 13,
            "SYM": 14,
            "VERB": 15,
            "X": 16,
            "z": 17}
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

s = "Apple is looking at buying U.K. startup for $1 billion"
doc = nlp(s)
pos_tags = [token.pos_ for token in doc]
print(pos_tags)
pos_tags += ["z" for i in range(40 - len(pos_tags))]
print(pos_tags)
pos_tags = pos_tags[:40]
print(pos_tags)
encoded_pos_tags = [tag_to_ix.get(tag, 17) for tag in pos_tags]  # Use 12 for unknown tags
print(encoded_pos_tags)
