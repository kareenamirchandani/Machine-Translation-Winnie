from transformers import MT5ForConditionalGeneration, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk
import torch
import evaluate
import numpy as np


model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
state = torch.load("weights.pth")
model.load_state_dict(state)

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

dataset = load_from_disk("SALT_SPLIT")

languages = dataset["train"].column_names
sourceLangauge = "Luganda"
targetLanguage = "English"
def select_language(data):
    data = data.remove_columns([l for l in languages if l != targetLanguage and l != sourceLangauge])
    data = data.rename_column(targetLanguage, "labels")
    data = data.rename_column(sourceLangauge, "input_ids")
    return data
dataset["test"] = select_language(dataset["test"])
dataset["train"] = select_language(dataset["train"])

max_input_length = 128
max_target_length = 128

def preprocess(examples):
    model_inputs = tokenizer(text=examples["input_ids"], text_target=examples["labels"], max_length=max_input_length, truncation=True, padding=False)
    return model_inputs

model_input = dataset.map(preprocess, batched=True)

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

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_input_length)

sacrebleu = evaluate.load("sacrebleu")

training_args = Seq2SeqTrainingArguments(
    output_dir = ("MT5_" + sourceLangauge),
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    weight_decay = 0.01,
    save_total_limit = 3,
    num_train_epochs = 20,
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

metrics = trainer.evaluate()

print(metrics)

evaluate.save("FinalEval", **metrics)
