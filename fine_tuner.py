import os
import torch
import time


from datetime import timedelta
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

load_dotenv()




model_name = os.getenv("MODEL_NAME")
model_name = os.getenv("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully")



def tokenize_data(examples):
    full_prompt = examples["prompt"] + "\n" + examples["completion"]
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    prompt_ids = tokenizer(examples["prompt"], truncation=True, max_length=512)["input_ids"]
    len_prompt = len(prompt_ids)

    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()
    labels[:len_prompt] = [-100] * len_prompt
    tokenized["labels"] = labels
    return tokenized


def load_data():
    dataset = load_dataset( "json", data_files="dataset.jsonl") 

    split_dataset = dataset["train"].train_test_split(test_size=0.2)

    train_dataset = split_dataset["train"]
    temp_dataset = split_dataset["test"].train_test_split(test_size=0.5)

    validation_dataset = temp_dataset["train"]
    test_dataset = temp_dataset["test"]
   
    tokenized_train = train_dataset.map(tokenize_data, remove_columns=train_dataset.column_names)
    tokenized_validation = validation_dataset.map(tokenize_data, remove_columns=validation_dataset.column_names)
    tokenized_test = test_dataset.map(tokenize_data, remove_columns=test_dataset.column_names)

    return tokenized_train, tokenized_validation, tokenized_test



# print(torch.cuda.is_available())



def load_model():
    model = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL_NAME"), device_map="auto", load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.1,
    bias = "none",
    task_type = "CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    return model


def get_trainer(model, tokenizer, training_args, train, val):

    data_collater = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    trainer = Trainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        data_collator=data_collater,
        train_dataset=train,
        eval_dataset= val
    )
    return trainer




if __name__== "__main__":
    train, val, test = load_data()


    model = load_model()
    training_args = TrainingArguments(
        output_dir="./hplc-mistral-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=".hplc-mistral-model/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        learning_rate=2e-4,
        label_names=["labels"],
        report_to="none",  
        fp16=True)
    
    trainer = get_trainer(model=model, tokenizer=tokenizer, training_args=training_args, train= train, val=val)
    
    start_time = time.time() 

    trainer.train() 

    end_time = time.time() 
    elapsed_time = end_time - start_time  
    formatted_time = str(timedelta(seconds=int(elapsed_time)))  
    print(f"Training time: {formatted_time}")  
    

    results = trainer.evaluate()
    print(results)  
    trainer.save_model("./hplc-mistral-model") 
    tokenizer.save_pretrained("./hplc-mistral-model")  

    print("Training complete. Model and tokenizer saved.")









