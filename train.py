import os
import re
import json
import yaml
import torch
from tqdm import tqdm
from transformers import TextStreamer

from datasets import Dataset

import unsloth
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig



#from transformers import TextStreamer

class finetunning():
    def __init__(self, epochs, data_path, data_path_test, device, max_seq_length=2048):
        self.epochs = epochs
        self.data_path = data_path
        self.data_path_test = data_path_test
        self.mlen = max_seq_length
        
        self.device = device

        file = open("params.json", "r")
        self.prm = json.load(file)
        file.close()

        self.HF = self.prm["hf_token"]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = self.prm["llm_name"], #"unsloth/Qwen3.5-4B",
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = True,
                    #load_in_16bit = True,
                    #full_finetuning = False,
                )
        
        self.model = FastLanguageModel.get_peft_model(
                        self.model,
                        
                        target_modules = self.prm["lora_modules"], # "all-linear", # Optional now! Can specify a list if needed
                        r = 16,           # The larger, the higher the accuracy, but might overfit
                        lora_alpha = 16,  # Recommended alpha == r at least
                        lora_dropout = 0,
                        bias = "none",
                        random_state = 67,
                        max_seq_length = max_seq_length,
                        use_rslora = False,  # We support rank stabilized LoRA
                        loftq_config = None, # And LoftQ
                        use_gradient_checkpointing = "unsloth", # Reduces memory usage
                )
        
    def outputer(self, data, lab, SS, SE):
        try:
            return f"{SS} {data[lab]} {SE}"
        except: # Test
            return False
    
    def get_data(self, prompt, name, label):
        file = open(prompt, "r", encoding="utf-8")
        self.system_prompt = yaml.safe_load(file)['prompt']
        file.close()

        #dataset = {"prompt": [], "answer": []}
        dataset = []
        SOLUTION_START = "<SOLUTION>"
        SOLUTION_END = "</SOLUTION>"

        with open(name, "r", encoding="utf-8") as f_in:
            for line in f_in:
                data = json.loads(line)

                if label == "span_annotations":
                    text_content = self.system_prompt.format(
                        language=data["lang"],
                        hope=data["primary_label"]
                    )
                else:
                    text_content = self.system_prompt.format(
                        language=data["lang"]
                    )

                input_sample = data["title"] + data["selftext"]

                the_output = self.outputer(data=data,lab=label, SS=SOLUTION_START, SE=SOLUTION_END)

                if the_output:
                    dict_prompt = {
                        "text": "<human>" +text_content+f"return your final answer between {SOLUTION_START} and {SOLUTION_END}" + input_sample + "\n<bot>" + the_output,
                        "metadata": {"lang": data["lang"]}
                        }
                    
                else:
                    dict_prompt = {
                        "text": "<human>" +text_content+f"return your final answer between {SOLUTION_START} and {SOLUTION_END}" + input_sample + "\n<bot>"
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

        #self.model.save_pretrained_gguf("model_repeater_spider_chunker", self.tokenizer,)
        #self.model.push_to_hub_gguf("JesusAS/model_repeater_spider_chunker", self.tokenizer, token = self.HF)

        
        FastLanguageModel.for_inference(self.model)

        #text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)

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
