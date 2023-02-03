from transformers import MT5ForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_dataset
import evaluate
import numpy as np
import json
import pynvml as pn

def print_gpu_utilization():
    pn.nvmlInit()
    handle = pn.nvmlDeviceGetHandleByIndex(0)
    info = pn.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_utilization()
    

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

### model has ~ 300,000,000 parameters ~2GB

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Load and dataset
dataset = load_dataset("Sunbird/salt-dataset", split = "train")
dataReduction = 1
dataset = dataset.select(list(range(0,int(dataReduction * dataset.num_rows))))
ttsplit = 0.8
dataset = dataset.train_test_split(train_size=ttsplit)

# Select a single language to train on
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


# Tokenize Input

max_input_length = 128
max_target_length = 128

def preprocess(examples):
    model_inputs = tokenizer(text=examples["input_ids"], text_target=examples["labels"], max_length=max_input_length, truncation=True, padding=False)
    return model_inputs

model_input = dataset.map(preprocess, batched=True)

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

# Set up own Custom training loop

learning_rate = 5e-5
per_device_train_batch_size = 64
per_device_eval_batch_size = 64
weight_decay = 0.01
num_train_epochs = 10
max_target_length = 128
loss_log_per_epoch = 5 # See loss_log_steps below datacollator

# Datacollators

def custom_collate(batch):
    model_inputs = {"input_ids":[torch.tensor(d["input_ids"]) for d in batch], "labels":[torch.tensor(d["labels"]) for d in batch]}
    model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
    model_inputs["labels"] = pad_sequence(model_inputs["labels"], batch_first=True, padding_value=-100)
    return model_inputs
train_dataloader = DataLoader(model_input["train"], shuffle=True, batch_size=per_device_train_batch_size, collate_fn=custom_collate)
eval_dataloader = DataLoader(model_input["test"], shuffle=True, batch_size=per_device_eval_batch_size, collate_fn=custom_collate)

loss_log_steps = int(float(len(train_dataloader))/float(loss_log_per_epoch))
# Optimizers

optimizer = AdamW(model.parameters(), lr = learning_rate)

# Progress bar

progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))

# Setup accelerator

accelerator = Accelerator()
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

# Training Loop

eval_data = []
train_loss_log = []
eval_loss_log = []
for epoch in range(num_train_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_dataloader): 
        labels = batch.pop("labels") # pop returns the value at associated key that was removed so batch is now {input_ids : [.[].[].]} and labels [.[].[].[].]
        loss, _ = model(**batch, labels = labels, use_cache=False)[:2] # _ is the logits however we will not use them, however they should should be predicted using generate
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if batch_idx % loss_log_steps == 0 and batch_idx > 0:
            train_loss_log.append((batch_idx, loss[0]))
            model.eval()
            eval_batch = next(iter(eval_dataloader))
            eval_labels = eval_batch.pop("labels")
            eval_loss, _ = model(**eval_batch, labels = eval_labels, use_cache=False)[:2]
            eval_loss_log.append((batch_idx, eval_loss[0]))
            model.train()
            
    model.eval()
    for batch in eval_dataloader:
        labels = batch.pop("labels")
        preds = model.generate(batch, max_length = max_target_length)
        results = compute_metrics((preds, labels))
        eval_data.append((epoch + 1, results))
    break

model.save_pretrained("MT5.2_" + sourceLangauge)

results = {"metrics": eval_data,
    "train_loss_log": train_loss_log,
    "eval_loss_log": eval_loss_log}

json_object = json.dumps(results)

with open("results.json", "w") as outfile:
    outfile.write(json_object)

model = accelerator.unwrap_model(model)
accelerator.save("MT5.2_" + sourceLangauge, model)
