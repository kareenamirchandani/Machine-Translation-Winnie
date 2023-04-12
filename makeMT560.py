# Make dataset from MT600 files

# Stage train.v1.lang, train.v1.eng.tok and train.v1.src.tok to $TMPDIR

from datasets import Dataset 
from tqdm.auto import tqdm

with open("train.v1.lang", "r") as f:
    langs = f.readlines()

with open("train.v1.eng.tok", "r") as f:
    eng = f.readlines()

with open("train.v1.src.tok", "r") as f:
    src = f.readlines()

langCodes = ["lug", "lgg", "ach", "teo", "nyn", "swa"]
languages = ["Lugandan", "Lugbara", "Acholi", "Ateso", "Runyankole", "Swahili"]
codeToLang = dict(zip(langCodes, languages))
langCodeIdx = {l: idx for idx, l in enumerate(langCodes)}
dataset = []

progress_bar = tqdm(range(len(langs)))

for idx, l in enumerate(langs):
    l = l.strip()
    if l in set(langCodes):
        dataset.append({})
        dataset[-1]["src"] = src[idx].strip()
        dataset[-1]["English"] = eng[idx].strip()
        dataset[-1]["src_lang"] = l
    progress_bar.update(1)

dataset = Dataset.from_list(dataset)

dataset = dataset.train_test_split(0.8)

dataset.save_to_disk("MT560")
