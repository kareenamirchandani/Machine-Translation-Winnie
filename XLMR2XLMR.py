from transformers import AutoModelForMaskedLM, EncoderDecoderModel, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, concatenate_datasets
import evaluate
import numpy as np


EncoderModel = "Davlan/afro-xlmr-mini"

AfroXLMR = AutoModelForMaskedLM.from_pretrained("Davlan/afro-xlmr-mini")

#### model has 117,891,474 parameters

tokenizer = AutoTokenizer.from_pretrained(EncoderModel)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(EncoderModel, EncoderModel)

# Accessing the model configuration

config_encoder = model.config.encoder

config_decoder = model.config.decoder

# set decoder config to causal lm

config_decoder.is_decoder = True

config_decoder.add_cross_attention = True

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# print(AfroXLMR.config)
# print(model.config)

## Will need to pre-train to ensure weights for new ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight'] are learnt
## Either use causal LM objective or BART denoising objective
## Am unsure what changing trainign objective from previous pre-training and adaptive fine tuning does
## Not sure if the XLMR encoder is best decoder for this scenario
## Should i use a GPT based one?? or BART based one??? T5 based one???
## Use same data as Afro-XLMR trained on

#########
# Training

# Load Dataset
dataset = load_dataset("Sunbird/salt-dataset", split = "train")
dataReduction = 1
dataset = dataset.select(list(range(0,int(dataReduction * dataset.num_rows))))
ttsplit = 0.8
dataset = dataset.train_test_split(train_size=ttsplit)

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(EncoderModel)

# Turn Dataset into Mul to En
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

# Tokenize Input
max_input_length = 128
max_target_length = 128

def preprocess(examples):
    model_inputs = tokenizer(examples["inputs"], max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["targets"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model_input = dataset.map(preprocess, batched=True, remove_columns=["inputs", "targets"]) ## Just inputs_ids and labels columns left

# DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=max_input_length)

# Evaluation Metric
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
    output_dir = "AfroXLMR-SALT.1",
    evaluation_strategy = "steps",
    eval_steps = 500,
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

trainer.save_model("finetunedXLMR")