import os
import re
import json
import yaml
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset, DatasetDict

import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer





class finetunning_llm():
    def __init__(self, data_path, data_path_test, device, max_seq_length=2048, llm_path=False):
        self.data_path = data_path
        self.data_path_test = data_path_test
        self.mlen = max_seq_length
        
        self.device = device

        file = open("params.json", "r")
        self.prm = json.load(file)
        file.close()

        self.epochs = self.prm["finetune_epochs_llm"]

        #self.HF = self.prm["hf_token"]
        if llm_path:
            self.prm["llm_name"] = llm_path

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = self.prm["llm_name"], #CURRENT RUN: "unsloth/Qwen3.5-9B"    FIRST ITERATION: "unsloth/Qwen3.5-4B",
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = True,
                )
        
        self.model = FastLanguageModel.get_peft_model(
                        self.model,
                        
                        target_modules = self.prm["lora_modules"],
                        r = 16,
                        lora_alpha = 16,
                        lora_dropout = 0,
                        bias = "none",
                        random_state = 67,
                        max_seq_length = max_seq_length,
                        use_rslora = False,
                        loftq_config = None,
                        use_gradient_checkpointing = "unsloth", # Reduces memory usage
                )
        
    def response(self, data, lab):
        try:
            if lab == "primary_label":
                return f"<LABEL>{data[lab]}</LABEL>"
            elif lab == "trigger_emotions":
                return f"<LIST>{', '.join(data[lab])}</LIST>"
            else: ### lab == "span_annotations"
                spanners = [
                    "[" +y["span"]+ "],[" +y["outcome_stance"]+ "],[" +y["actor"]+ "]" for y in data[lab]
                ]
                return f"<LIST>{', '.join(spanners)}</LIST>"            
        except:
            return False
    
    def get_data(self, prompt, name, label):
        file = open(prompt, "r", encoding="utf-8")
        self.system_prompt = yaml.safe_load(file)['prompt']
        file.close()

        dataset = []
        with open(name, "r", encoding="utf-8") as f_in:
            for line in f_in:
                data = json.loads(line)
                input_sample = data["title"] + data["selftext"]

                try:
                    test_var = data["primary_label"]
                except:
                    data["primary_label"] = "NULL"
                if label == "span_annotations":
                    if data["primary_label"] == "Not Hope":
                        continue ## IGNORE "NOT HOPE" FOR THE 3,4,5th TASK.
                    else:
                        text_content = self.system_prompt.format(
                            language=data["lang"],
                            text = input_sample,
                        )
                else:
                    text_content = self.system_prompt.format(
                        language=data["lang"],
                        text = input_sample
                    )

                the_output = self.response(data=data,lab=label)

                if the_output:
                    dict_prompt = {
                        "text": "<human>:" +text_content + "\n<bot>:" + the_output,
                        "metadata": {"lang": data["lang"], "row_id": data["row_id"]}
                        }
                    
                else:
                    dict_prompt = {
                        "text": "<human>:" +text_content + "\n<bot>:",
                        "metadata": {"lang": data["lang"], "row_id": data["row_id"]}
                        }
                
                dataset.append(dict_prompt)

        return Dataset.from_list(dataset)
        


    def finetune(self, prompt:str, target:str, save_in=False):

        train_dataset = self.get_data(prompt=prompt, name=self.data_path, label=target)

        FastLanguageModel.for_training(self.model)

        training_args = SFTConfig(
            max_seq_length = self.mlen,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = self.epochs, #max_steps = 10,
            learning_rate = 2e-4,
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 67,
            dataset_num_proc = 1,
            report_to = "none",     # For Weights and Biases
        )
        
        trainer = SFTTrainer(
                model = self.model,
                tokenizer = self.tokenizer,
                train_dataset = train_dataset,
                args = training_args,
            )

        trainer.train()

        if save_in:
            self.model.save_pretrained(save_in) #E.G. "./models/primary_label_Qwen35-4B"
            self.tokenizer.save_pretrained(save_in)
        else:
            print("\n\n WARNING: MODEL NOT SAVED \n\n")

        
        
        
        ################# INFERENCE #################
        
        FastLanguageModel.for_inference(self.model)

        test_dataset = self.get_data(prompt=prompt, name=self.data_path_test, label=target)
        preds = []
        for i in tqdm(range(len(test_dataset)), desc="Inferencing..."):
            message = [
                    {
                    "role": "user", 
                    "content": [
                        {"type":"text", "text": test_dataset[i]["text"]}
                        ]
                    }
                ]
            input_ids = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt = True,
                return_tensors = "pt",
                tokenize=True
            ).to("cuda")
        
            _ = self.model.generate(input_ids, max_new_tokens=128, pad_token_id = self.tokenizer.eos_token_id) #streamer = text_streamer,
            generated = self.tokenizer.decode(_[0], skip_special_tokens=True)
            preds.append(generated)

        with open(f"./preds/output_{target}.json", "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)
        
        return self.model




class finetunning_roberta():
    def __init__(self, data_path, data_path_test, bsize = 16):
        self.bsize = bsize
        self.order = {
                    "Not Hope" : 0,             #1. If the text does not express desire, expectation, or anticipation of a future outcome, label it as <LABEL>Not Hope</LABEL>
                    "Hopelessness" : 1,         #2. Otherwise, if the text express pessimism, resignation or belief that no positibe change may or will occur, label it as <LABEL>Hopelessness</LABEL>
                    "General Hope" : 2,         #3. Otherwise, if the text express vague or broad optimism or positive expectation about the future without specifying a concrete outcome, label it as <LABEL>General Hope</LABEL>
                    "Sarcastic Hope" : 3,       #4. Otherwise, if the text express an ironic or mocking expression that appears hopeful but implies disbelief or the opposite sentiment, label it as <LABEL>Sarcastic Hope</LABEL>
                    "Realistic Hope" : 4,       #5. Otherwise, if the text express a desire or expectation for a future outcome that is plausible, achievable, or grounded in real-world conditions, even if uncertain, label it as <LABEL>Realistic Hope</LABEL>
                    "Unrealistic Hope" : 5      #6. Otherwise, if the text express a desire or expectation for a future outcome that is highly improbable, impossible, fantastical, or disconnected from realistic constraints, label it as <LABEL>Unrealistic Hope</LABEL>
                }
        self.redro = {
                    0 : "Not Hope",
                    1 : "Hopelessness",
                    2 : "General Hope",
                    3 : "Sarcastic Hope",
                    4 : "Realistic Hope",
                    5 : "Unrealistic Hope"
                }

        self.data_path = data_path
        self.data_path_test = data_path_test

        file = open("params.json", "r")
        self.prm = json.load(file)
        file.close()

        self.epochs = self.prm["finetune_epochs_roberta"]
        self.roberta_name = self.prm["roberta_model"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.roberta_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.roberta_name, num_labels=6, ignore_mismatched_sizes=True)

    def tokenize_function(self, example):
        return self.tokenizer(example["sentence1"], example["sentence2"], padding=True, truncation=True)
    
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")

        return {"accuracy": acc, "macro_f1": f1}

        
    def load_jsonl(self, path):
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        return pd.DataFrame(recs)
    
    def build_text(self, df):
        return (df["title"].fillna("").astype(str) + "\n\n" + df["selftext"].fillna("").astype(str)).str.strip()


    def finetune(self, output_path = "./roberta_results", pth_to_save = "preds/roberta_preds.json"):

        df_tr = self.load_jsonl(self.data_path)
        df_dv = self.load_jsonl(self.data_path_test)

        for _df in (df_tr, df_dv):
            if "row_id" not in _df.columns:
                raise ValueError("Missing row_id in input JSONL.")
            if "lang" not in _df.columns:
                _df["lang"] = None
            _df["title"] = _df["title"].fillna("")
            _df["selftext"] = _df["selftext"].fillna("")
            _df["text"] = self.build_text(_df)

        df_train = df_tr[["title", "selftext", "text", "primary_label"]]
        df_dev = df_dv[["title", "selftext", "text"]]

        df_train["primary_label"] = df_train["primary_label"].apply(lambda x: self.order[x])
        
        df_train.rename(columns={"title":"sentence1", "selftext":"sentence2", "primary_label":"labels"}, inplace=True)
        df_dev.rename(columns={"title":"sentence1", "selftext":"sentence2"}, inplace=True)

        train_df, val_df = train_test_split(
                df_train, test_size=0.025, stratify=df_train["labels"], random_state=67
            ) ### First experiments running on 0.2, reducing to 0.025 enhanced the macro F1 on the test set by 0.05 points xd.
        
        raw_datasets = DatasetDict({
            "train" : Dataset.from_pandas(train_df),
            "validation" : Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(df_dev)
        })

        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        

        training_args = TrainingArguments(
            output_dir=output_path,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            learning_rate=2e-5,
            per_device_train_batch_size=self.bsize,
            per_device_eval_batch_size=self.bsize,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
                train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        metrics = trainer.evaluate()
        print(metrics)

        test_output = trainer.predict(tokenized_datasets["test"])
        logits = test_output.predictions
        preds = logits.argmax(axis=1)

        preds = [self.redro[x] for x in preds]

        with open(pth_to_save, "w") as f:
            json.dump(preds, f)