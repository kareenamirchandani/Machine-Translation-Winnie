from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets
import evaluate
import numpy as np

dataset = load_dataset("Sunbird/salt-dataset", split = "train")
dataReduction = 0.1
dataset = dataset.select(list(range(0,int(dataReduction * dataset.num_rows))))
ttsplit = 0.8
dataset = dataset.train_test_split(train_size=ttsplit)

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt", tgt_lang = "en_XX", src_lang = "te_IN") # Will let all ugandan languages be "te_IN"
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
model = MBartForConditionalGeneration.from_pretrained("AfroXLMR-SALT.1/checkpoint-1000")

#### Turn dataset into a mul to english dataset. Dataset with two columns; English and Mul, where mul is all the ugandan languages

def mix_langauges(data):
    languages = data.column_names
    langDataSets = {}

    for lang in languages:
        if lang != "English":
            langDataSets[lang] = data.remove_columns([l for l in languages if l != lang and l != "English" ])
            langDataSets[lang] = langDataSets[lang].rename_column(lang, "inputs")
            langDataSets[lang] = langDataSets[lang].rename_column("English", "targets")

    data = concatenate_datasets(list(langDataSets.values()))
    data = data.shuffle(seed=843) ## Shuffle new dataset so all languages are mixed up
    return data

dataset["train"] = mix_langauges(dataset["train"])
dataset["test"] = mix_langauges(dataset["test"])

### Tokenize the input with target English words along side as targets

max_input_length = 128
max_target_length = 128

def preprocess(examples):
    model_inputs = tokenizer(examples["inputs"], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["targets"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model_input = dataset.map(preprocess, batched=True, remove_columns=["inputs", "targets"]) ## Just inputs_ids and labels columns left

#### This will pad the inputs and outputs as the model is trained
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_input_length)

#### Will give evaluation metrics as model is trained

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
    

training_args = Seq2SeqTrainingArguments(
    output_dir = "mBART-SALT.1",
    evaluation_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    weight_decay = 0.01,
    save_total_limit = 3,
    num_train_epochs = 2,
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

trainer.train()



# src = "Hello World this is a script to translate"

# src_token = tokenizer(src, return_tensors="pt")

# out_token = model.generate(**src_token)

# out = tokenizer.batch_decode(out_token)

# print(out_token)

# print(out)



