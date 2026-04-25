from train import finetunning

import torch
import json

#"openai-community/gpt2"

if __name__ == "__main__":
    ft = finetunning(
            data_path = "./data/HopeEXP_Train.jsonl",
            data_path_test = "./data/HopeEXP_Test_unlabeled.jsonl",
            epochs = 3,
            device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    
    x = "trigger_emotions"
    print(f"\n\n############################################## {x} ##############################################")
    model = ft.finetune(prompt=f"./prompts/prompt_{x}.yml", target=x)

    x = "primary_label"
    print(f"\n\n############################################## {x} ##############################################")
    model = ft.finetune(prompt=f"./prompts/prompt_{x}.yml", target=x)

    x = "span_annotations"
    print(f"\n\n############################################## {x} ##############################################")
    model = ft.finetune(prompt=f"./prompts/prompt_{x}.yml", target=x)
