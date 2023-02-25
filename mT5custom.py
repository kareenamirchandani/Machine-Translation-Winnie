from transformers import MT5ForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_from_disk
import evaluate
import numpy as np
import json
import os
import shutil

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

### model has ~ 300,000,000 parameters ~2GB

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Load and dataset
dataset = load_from_disk("SALT_SPLIT")

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

def save_results_to_json(outputDir, eval_data, train_loss_log):
    results = {"metrics": eval_data,
        "train_loss_log": train_loss_log}

    json_object = json.dumps(results)

    with open((outputDir), "w") as outfile:
        outfile.write(json_object)

# Set up own Custom training loop

learning_rate = 5e-5
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
weight_decay = 0.01
num_train_epochs = 100
max_target_length = 128
epoch_per_save = 5
loss_log_per_epoch = 5 # See loss_log_steps below datacollator !!!! temp: Will overwrite loss_log_steps with fixed value
max_num_checkpoints = 1

# Datacollators

datacoll = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

def custom_collate(batch):
    model_inputs = {"input_ids":[torch.tensor(d["input_ids"]) for d in batch], "labels":[torch.tensor(d["labels"]) for d in batch], "attention_mask":[torch.tensor(d["attention_mask"]) for d in batch]}
    model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
    model_inputs["labels"] = pad_sequence(model_inputs["labels"], batch_first=True, padding_value=-100)
    model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], batch_first=True, padding_value=0)
    return model_inputs
train_dataloader = DataLoader(model_input["train"], shuffle=True, batch_size=per_device_train_batch_size, collate_fn=custom_collate)
eval_dataloader = DataLoader(model_input["test"], shuffle=True, batch_size=per_device_eval_batch_size, collate_fn=custom_collate)

loss_log_steps = int(float(len(train_dataloader))/float(loss_log_per_epoch))
loss_log_steps = 100
# Optimizers

optimizer = AdamW(model.parameters(), lr = learning_rate)

# Progress bar

progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))

# Setup accelerator

accelerator = Accelerator()
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

# Checkpointing
version = ""
outputDir = "MT5_" + sourceLangauge + version
checkpointDir = outputDir + "/Checkpoints"
os.mkdir(outputDir)
os.mkdir(checkpointDir)

# Training Loop

eval_data = [] 
train_loss_log = []
eval_loss_log = []
step = 0
for epoch in range(num_train_epochs):
    model.train()
    total_train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader): 
        labels = batch.pop("labels") # pop returns the value at associated key that was removed so batch is now {input_ids : [.[].[].]} and labels [.[].[].[].]
        loss, _ = model(**batch, labels = labels, use_cache=False)[:2] # _ is the logits however we will not use them, however they should should be predicted using generate
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        total_train_loss += loss.item()
        step += 1
        if batch_idx % loss_log_steps == 0 and batch_idx > 0:
            train_loss_log.append((step, total_train_loss/loss_log_steps))  
            total_train_loss = 0
    model.eval()
    eval_data.append({ "bleu": 0, "loss": 0, "step": step})
    for batch in eval_dataloader:
        labels = batch.pop("labels")
        # if hasattr(model, "generate"):   ### For some reason on GPU model.generate doesn't work so will just use loss instead
        #     with torch.no_grad():
        #         preds = model.generate(**batch, max_length = max_target_length)
        #     preds = accelerator.gather(preds)
        #     labels = accelerator.gather(labels)
        #     results = compute_metrics((preds, labels))
        #     eval_data[-1]["bleu"] += results["bleu"]
        with torch.no_grad():
            loss, _ = model(**batch, labels = labels, use_cache=False)[:2]
        loss = accelerator.gather(loss)
        eval_data[-1]["loss"] += loss.item()

    for k,v in eval_data[-1].items():
        eval_data[-1][k] = eval_data[-1][k]/len(eval_dataloader)

    accelerator.print(eval_data[-1])

    if epoch % epoch_per_save == 0 and epoch > 1:
        checkpoints = [int(check) for check in os.listdir(checkpointDir)]
        if len(checkpoints) >= max_num_checkpoints:
            checkpoints.sort()
            shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
        accelerator.save_state(output_dir=(checkpointDir + "/" + str(epoch)))
        save_results_to_json((checkpointDir + "/" + str(epoch) + "/results.json"), eval_data, train_loss_log)

checkpoints = [int(check) for check in os.listdir(checkpointDir)]
if len(checkpoints) >= max_num_checkpoints:
    checkpoints.sort()
    shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
accelerator.free_memory()
unwrapped = accelerator.unwrap_model(model=model)
accelerator.save(unwrapped.state_dict(), (outputDir + "/weights.pth"))
save_results_to_json((outputDir + "/Finalresults.json"), eval_data, train_loss_log)

#### Why Does this perform so much worse

### Expand data to MT560 and FLORES and include back translation
### Train a multilingual model then fine-tune on specific languages
### Include gradient accumulation