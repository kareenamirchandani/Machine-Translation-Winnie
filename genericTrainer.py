from transformers import MT5ForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration, MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForSeq2SeqLM, MT5TokenizerFast, get_cosine_schedule_with_warmup, get_constant_schedule
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_from_disk, concatenate_datasets, Dataset
import evaluate
import numpy as np
import json
import os
import shutil
import sys
import random
from math import floor

def create_dataset(datasets: list, datasetNames: list, config):
    def sample(ds, sf):  # This will upsample and downsample SALT in the case of MUL
        if sf == 0:
            ds = ds.filter(lambda ex: False)
        elif sf < 1:
            ds = ds.shuffle().shard(num_shards = 1/sf, index = 0)
        elif sf > 1:
            ds = [ds] * floor(sf) 
            if sf % 1 != 0:
                ds.append(ds[0].shuffle().shard(num_shards = 1/(sf - floor(sf))))
            ds = concatenate_datasets(ds)
        return ds
    
    dl = datasets[0]
    dl = dl.filter(lambda ex: False)
    for dataset, name in zip(datasets, datasetNames):
        for split in dataset.column_names:
            datasetList = []
            ds = dataset[split]
            if config["source_language"] == "mul" or split == "test":
                languages = config["language_codes"]
            else:
                languages = [config["source_language"]]
            for lang in languages:
                dsFilt = ds.filter(lambda ex: ex["src_lang"] == lang)
                if split == "train":
                    dsSampled = sample(dsFilt, config["sampleFactor"][name][lang])
                elif split == "test":
                    dsSampled = sample(dsFilt, config["eval_sampleFactor"][name][lang])
                datasetList.append(dsSampled)
            dt = concatenate_datasets(datasetList)
            dt = dt.rename_column("src", "input_ids")
            dt = dt.rename_column("English", "labels")
            dl[split] = concatenate_datasets([dl[split], dt])
    return dl

def save_results_to_json(train_loss, eval_loss, bleu, dir):
    results = {"train_loss": train_loss,
            "eval_loss": eval_loss,
            "bleu": bleu}
    json_object = json.dumps(results)

    with open(dir, "w") as outfile:
        outfile.write(json_object)

def train(model, tokenizer, dataset, config):

    print(dataset)

    # Tokenize Input

    max_input_length = 128
    max_target_length = 128

    def preprocess(examples):
        model_inputs = tokenizer(text=examples["input_ids"], text_target=examples["labels"], max_length=max_input_length, truncation=True, padding=False)
        return model_inputs

    model_input = dataset["train"].map(preprocess, batched=True)
    eval_input = {lang: dataset["test"].filter(lambda ex: ex["src_lang"] == lang).map(preprocess, batched=True) for lang in config["language_codes"]} #Eval Dataset for each language

    # Functions for generating BLEU score from predictions

    sacrebleu = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    # Set up own Custom training loop
    
    num_train_epochs = config["training_epochs"]

    learning_rate = 5e-5
    per_device_train_batch_size = config["batch_size"]
    per_device_eval_batch_size = config["batch_size"]
    gradient_accumulation = 10
    epoch_per_save = 5
    epoch_per_eval = 4
    loss_log_steps = 100
    max_num_checkpoints = 5

    def custom_collate(batch):
        model_inputs = {"input_ids":[torch.tensor(d["input_ids"]) for d in batch], "labels":[torch.tensor(d["labels"]) for d in batch], "attention_mask":[torch.tensor(d["attention_mask"]) for d in batch]}
        model_inputs["input_ids"] = pad_sequence(model_inputs["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
        model_inputs["labels"] = pad_sequence(model_inputs["labels"], batch_first=True, padding_value=-100)
        model_inputs["attention_mask"] = pad_sequence(model_inputs["attention_mask"], batch_first=True, padding_value=0)
        return model_inputs
    train_dataloader = DataLoader(model_input, shuffle=True, batch_size=per_device_train_batch_size, collate_fn=custom_collate)
    eval_dataloader = {lang: DataLoader(eval_input[lang], shuffle=True, batch_size=per_device_eval_batch_size, collate_fn=custom_collate) for lang in config["language_codes"]} #Eval Dataloader for each language
    # Optimizers

    optimizer = AdamW(model.parameters(), lr = learning_rate)

    # Learning Rate Scheduler

    #lrSchedule = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_epochs*len(train_dataloader))
    lrSchedule = get_constant_schedule(optimizer=optimizer) # Will use constant schedule for now. Change if significant noise in loss graphs


    # Progress bar

    progress_bar = tqdm(range(num_train_epochs * len(train_dataloader)))

    # Setup accelerator

    eval_dataloader_list = [eval_dataloader[lang] for lang in config["language_codes"]] # Many Eval_dataloaders so must be handled carefully

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation)
    acceleratedComponents = accelerator.prepare(model, optimizer, lrSchedule, train_dataloader, *eval_dataloader_list)

    model = acceleratedComponents[0]
    optimizer = acceleratedComponents[1]
    lrSchedule = acceleratedComponents[2]
    train_dataloader = acceleratedComponents[3]
    eval_dataloader_list = acceleratedComponents[4:]
    eval_dataloader = {lang: eval_dataloader_list[idx] for idx, lang in enumerate(config["language_codes"])}

    # Checkpointing
    outputDir = config["output_dir"] + config["source_language"]
    checkpointDir = outputDir + "/Checkpoints"
    os.mkdir(outputDir)
    os.mkdir(checkpointDir)

    # Training Loop
 
    train_loss_log = []
    bleu_data = {lang: [] for lang in config["language_codes"]}
    eval_data = {lang: [] for lang in config["language_codes"]}
    step = 0
    eval_in_decline = False
    for epoch in range(num_train_epochs):
        model.train()
        total_train_loss = 0
        batch_idx = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model): 
                optimizer.zero_grad()
                labels = batch.pop("labels") # pop returns the value at associated key that was removed so batch is now {input_ids : [.[].[].]} and labels [.[].[].[].]
                loss, _ = model(**batch, labels = labels, use_cache=False)[:2] # _ is the logits however we will not use them, however they should should be predicted using generate
                accelerator.backward(loss)
                optimizer.step()
                lrSchedule.step()
            progress_bar.update(1)
            total_train_loss += loss.item()
            if batch_idx % loss_log_steps == 0 and batch_idx > 0:
                train_loss_log.append((step, total_train_loss/loss_log_steps))  
                total_train_loss = 0
            batch_idx += 1
            step += 1
        
        model.eval()
        for lang in config["language_codes"]:
            eval_data[lang].append(0)
            for batch in eval_dataloader[lang]:
                labels = batch.pop("labels")
                with torch.no_grad():
                    loss, _ = model(**batch, labels = labels, use_cache=False)[:2]
                loss = accelerator.gather(loss)
                eval_data[lang][-1] += loss.item()
            eval_data[lang][-1] = (step, eval_data[lang][-1]/len(eval_dataloader[lang]))
        
        # if len(eval_data) >= 2: !!!! Needs to be edited for loss calculated individuallt for each language
        #     if eval_data[-1] > eval_data[-2]:
        #         print("Evaluation Performing Worse. Final Model will not be saved only best performing one")
        #         eval_in_decline = False # Changed True to False as early stopping is causing errors

        if epoch % epoch_per_eval == 0: #and epoch > 1
            for lang in config["language_codes"]:
                sacrebleu = evaluate.load("sacrebleu")
                for batch in eval_dataloader[lang]:
                    labels = batch.pop("labels")
                    with torch.no_grad():
                        preds = accelerator.unwrap_model(model).generate(batch["input_ids"], attention_mask=batch["attention_mask"], max_length = max_target_length)
                        preds = accelerator.gather(preds).cpu().numpy()
                        labels = accelerator.gather(labels).cpu().numpy()

                        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                        
                        labels, preds = postprocess_text(preds, labels)

                        sacrebleu.add_batch(predictions=preds, references=labels)
                metrics = sacrebleu.compute()
                bleu_data[lang].append((step, metrics["score"]))
                print(bleu_data[lang], lang)


        if epoch % epoch_per_save == 0 and epoch > 1 and eval_in_decline == False:
            checkpoints = [int(check) for check in os.listdir(checkpointDir)]
            if len(checkpoints) >= max_num_checkpoints:
                checkpoints.sort()
                shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
            accelerator.save_state(output_dir=(checkpointDir + "/" + str(epoch)))

        if eval_in_decline:
            break

    checkpoints = [int(check) for check in os.listdir(checkpointDir)]
    checkpoints.sort()
    if eval_in_decline:
        accelerator.load_state(checkpointDir + "/" + str(checkpoints[-2]))
        if len(checkpoints) >= max_num_checkpoints:
            shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
        accelerator.free_memory()
        unwrapped = accelerator.unwrap_model(model=model)
        accelerator.save(unwrapped.state_dict(), (outputDir + "/weights.pth"))
    else:
        if len(checkpoints) >= max_num_checkpoints:
            shutil.rmtree(checkpointDir + "/" + str(checkpoints[0]))
        accelerator.free_memory()
        unwrapped = accelerator.unwrap_model(model=model)
        accelerator.save(unwrapped.state_dict(), (outputDir + "/weights.pth"))
    
    save_results_to_json(train_loss_log, eval_data, bleu_data, (outputDir + "/results.json"))

def eval(model, tokenizer, eval_dataset, config):
    max_input_length = 128

    def preprocess(examples):
        model_inputs = tokenizer(text=examples["input_ids"], text_target=examples["labels"], max_length=max_input_length, truncation=True, padding=False)
        return model_inputs

    # model_input = eval_dataset.map(preprocess, batched=True)

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
        output_dir = ("MT5_"),
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = config["batch_size"],
        weight_decay = 0.01,
        save_total_limit = 3,
        num_train_epochs = 20,
        predict_with_generate = True,
    )
    bleuScores = dict.fromkeys(config["language_codes"])
    for lang in config["language_codes"]:
        eval_dataset = eval_dataset.filter(lambda ex: ex["src_lang"] == lang)
        eval_dataset = eval_dataset.map(preprocess, batched=True)

        trainer = Seq2SeqTrainer(
            model = model,
            args = training_args,
            train_dataset = eval_dataset,
            eval_dataset = eval_dataset,
            tokenizer = tokenizer,
            data_collator = data_collator,
            compute_metrics = compute_metrics
        )
        metrics = trainer.evaluate()
        bleuScores[lang] = metrics

    with open((config["output_dir"] + config["source_language"] + "/results.json"), "r") as f:
        results = json.load(f)
    results["bleu"] = bleuScores
    with open((config["output_dir"] + config["source_language"] + "/results.json"), "w") as f:
        json.dumps(results, f)

def model_from_weights(model, weight_path):
    state = torch.load(weight_path)
    model.load_state_dict(state)
    return model

def add_language_tokens(tokenizer, tokens_to_replace, language_tokens):
    for old, new in zip(tokens_to_replace, language_tokens):
        if (">>" + new + "<<") not in tokenizer.encoder and old in tokenizer.encoder:
            tokenizer.encoder[(">>" + new + "<<")] = tokenizer.encoder[old]
            del tokenizer.encoder[old]
    return tokenizer

args = sys.argv
possibleArgs = set(["mt5-small", "byt5-small","mt5-base", "byt5-base", "mt5-large", "byt5-large", "mbart", "opus"])
langaugeArgs = {1:"lug", 2:"lgg", 3:"ach", 4:"teo", 5:"nyn", 6:"swa"} # this will be passed in by $PBS_ARRAY_INDEX
arg = 0
langArg = 0
for a in args:
    if a in possibleArgs:
        arg = a
    if a in langaugeArgs:
        langArg = langaugeArgs[a]
if arg == 0:
    arg = "opus" # Defaults to opus if no arguement given
if langArg == 0:
    langArg = "mul"

config = {"source_language": langArg, # "Give as code e.g. lug lgg, teo"
        "append_language_tokens": False,
        "language_codes": ["lug", "lgg", "ach", "teo", "nyn", "swa"],   # Add functionality to add Language tokens to the start of each sentence      
        "languages": ["Luganda", "Lugbara", "Acholi", "Ateso", "Runyankole", "Swahili"],
        "langToCode": {"Luganda": "lug", "Lugbara": "lgg", "Acholi": "ach", "Ateso":"teo", "Runyankole":"nyn", "Swahili": "swa"},
        "sampleFactor": {"SALT": {"lug": 1, "lgg": 1, "teo": 1, "nyn": 1, "swa": 1, "ach": 1}, "MT560": {"lug": 1, "lgg": 1, "teo": 1, "nyn": 1, "swa": 1, "ach":1}}, # Change how much of each language to use from each dataset
        "eval_sampleFactor": {"SALT": {"lug": 1, "lgg": 1, "teo": 1, "nyn": 1, "swa": 1, "ach": 1}, "MT560": {"lug": 0, "lgg": 0, "teo": 0, "nyn": 0, "swa": 1, "ach":0}},
        "training_epochs": 20,
        "output_dir": (arg + "_"),
        "batch_size": 32
}

datasetSALT = load_from_disk("SALT_SPLIT")

datasetMT560 = load_from_disk("MT560")

# datasetMT560 = Dataset.from_list([{"src": "Hello World", "English": "Bye World", "src_lang": "swa"}, {"src": " World Hello", "English": "  World Bye", "src_lang": "lug"}])
# datasetMT560 = datasetMT560.train_test_split(0.5)
dataset = create_dataset([datasetSALT, datasetMT560],["SALT", "MT560"], config)

if arg == "mt5-small":
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-small")
elif arg == "byt5-small":
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
    tokenizer = MT5TokenizerFast.from_pretrained("google/byt5-small")
elif arg == "mt5-base":
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-base")
elif arg == "byt5-base":
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-base")
    tokenizer = MT5TokenizerFast.from_pretrained("google/byt5-base")
elif arg == "mt5-large":
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")
    tokenizer = MT5TokenizerFast.from_pretrained("google/mt5-large")
elif arg == "byt5-large":
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-large")
    tokenizer = MT5TokenizerFast.from_pretrained("google/byt5-large")
elif arg == "mbart":
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt", tgt_lang="en_XX")
elif arg == "opus":
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    # random.seed(3421)
    # old_token_ids = [random.randint(1000,5000) for _ in config["language_codes"]]
    # old_tokens = tokenizer.convert_ids_to_tokens(old_token_ids)
    # tokenizer = add_language_tokens(tokenizer, old_tokens, config["language_codes"]) # Add_language_tokens will only work for OPUS as marian tokenizer has method encoder

dirs = os.listdir()
for f in dirs:
    if f == "weights.pth":
        print("Using weights from $TMPDIR/weights.pth")
        model = model_from_weights(model, "weights.pth") # If a weight file is loaded into the working directory the model will be based off that

train(model, tokenizer, dataset, config)