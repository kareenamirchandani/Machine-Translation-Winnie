from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_from_disk
from tqdm.auto import tqdm
import string

tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
codes = set(["B-PER", "B-ORG", "B-LOC", "I-PER", "I-ORG", "I-LOC"])

data = load_from_disk("SALT_SPLIT")

comparison = []

# bar = tqdm(range(len(data["train"])))

for ex in data["train"]:
    eng = ex["English"].split()
    src = ex["src"].split()
    namEnt = set(eng).intersection(set(src))
    pos = [idx for idx, x in enumerate(src) if x in namEnt]

    if not namEnt:
        continue

    output = nlp(ex["src"])
    tokSrc = tokenizer.encode(ex["src"])
    decSrc = tokenizer.batch_decode(tokSrc, clean_up_tokenization_spaces=False)
    posAfro = []
    for nE in output:
        if nE["entity"] in codes:
            word = ex["src"][nE["start"]:nE["end"]]
            idx = nE["end"]
            while True:
                if ex["src"][idx] == " " or ex["src"][idx] in string.punctuation:
                    break
                else:
                    word += ex["src"][idx]
                    idx += 1
            print(word)
            noPunc = ex["src"].translate(str.maketrans('', '', string.punctuation)).split()
            try:
                posAfro.append(noPunc.index(word))
            except:
                idx += 0
    print(ex["English"], ex["src"])
    print((pos, posAfro))
    comparison.append((pos, posAfro))
