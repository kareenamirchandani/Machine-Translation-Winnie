from transformers import MT5ForConditionalGeneration, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import numpy as np

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

### model has ~ 300,000,000 parameters ~2GB

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Load and dataset
dataset = load_dataset("Sunbird/salt-dataset", split = "train")
dataReduction = 1
dataset = dataset.select(list(range(0,int(dataReduction * dataset.num_rows))))
ttsplit = 0.8
dataset = dataset.train_test_split(train_size=ttsplit)

# def mix_langauges(data):
#     languages = data.column_names
#     langDataSets = {}

#     for lang in languages:
#         if lang != "English":
#             langDataSets[lang] = data.remove_columns([l for l in languages if l != lang and l != "English" ])
#             langDataSets[lang] = langDataSets[lang].rename_column(lang, "inputs")
#             langDataSets[lang] = langDataSets[lang].rename_column("English", "targets")

#     data = concatenate_datasets(list(langDataSets.values()))
#     data = data.shuffle(seed=843) ## Shuffle new dataset so all languages are mixed up
#     return data

# dataset["train"] = mix_langauges(dataset["train"])
# dataset["test"] = mix_langauges(dataset["test"])

# Select a single language to train on
languages = dataset["train"].column_names
print(languages)
sourceLangauge = "Luganda"
targetLanguage = "English"
def select_language(data):
    data = data.remove_columns([l for l in languages if l != targetLanguage and l != sourceLangauge])
    data = data.rename_column(targetLanguage, "targets")
    data = data.rename_column(sourceLangauge, "inputs")
    return data
dataset["test"] = select_language(dataset["test"])
dataset["train"] = select_language(dataset["train"])


# Tokenize Input

max_input_length = 128
max_target_length = 128

def preprocess(examples):
    model_inputs = tokenizer(examples["inputs"], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["targets"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model_input = dataset.map(preprocess, batched=True, remove_columns=["inputs", "targets"])

# DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_input_length)

# Evaluate
sacrebleu = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) ## where ID is -100 replace it with a pad token instead (think outputs are paded with -100 rather than <pad>)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels) ## Strips all predictions and labels

    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": result["score"]}

# Setup for training
training_args = Seq2SeqTrainingArguments(
    output_dir = ("MT5_" + sourceLangauge),
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64,
    weight_decay = 0.01,
    save_total_limit = 3,
    num_train_epochs = 5,
    predict_with_generate = True,
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = model_input["train"],
    eval_dataset = model_input["test"],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics
)

# Train
trainer.train()

trainer.save_model("MT5.2_" + sourceLangauge)

metrics = trainer.evaluate()

evaluate.save("FinalEval", metrics)

