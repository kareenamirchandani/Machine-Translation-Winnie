# from datasets import load_dataset, concatenate_datasets

# dataset = load_dataset("Sunbird/salt-dataset", split="train")

# def select_language(data, targetLanguage, sourceLangauge):
#     langToCode = {"Luganda": "lug", "Lugbara": "lgg", "Acholi": "ach", "Ateso":"teo", "Runyankole":"nyn", "Swahili": "swa"}
#     languages = data.column_names
#     if sourceLangauge in set(data.column_names):
#         data = data.remove_columns([l for l in languages if l != targetLanguage and l != sourceLangauge])
#         # data = data.rename_column(targetLanguage, "English")
#         data = data.rename_column(sourceLangauge, "src")
#         src_lang = [langToCode[lang]] * len(data)
#         data = data.add_column("src_lang", src_lang)
#     return data

# langDatasets = []
# for lang in dataset.column_names:
#     if lang != "English":
#         dt = select_language(dataset, "English", lang)
#         print(dt)
#         langDatasets.append(dt)

# dataset = concatenate_datasets(langDatasets)

# dataset = dataset.train_test_split(train_size=0.8)
# dataset.save_to_disk("SALT_SPLIT")

a = input("Enter Number to Factorial: ")