from train import finetunning_llm, roberta
from generate_submissions import this_is_the_most_important_function
import torch
import json

if __name__ == "__main__":

########################################################## RoBERTa for your usual classification task.
        
    RoBERTa_hope = roberta(
            data_path = "./data/HopeEXP_Train.jsonl",
            data_path_test = "./data/HopeEXP_Test_unlabeled.jsonl"
        )

    x = "primary_label"
    print(f"\n\n############################################## {x} ##############################################")
    model = RoBERTa_hope.finetune()


########################################################## And Qwen3.5 for the other tasks :)

    ft_emos = finetunning_llm(
            data_path = "./data/HopeEXP_Train.jsonl",
            data_path_test = "./data/HopeEXP_Test_unlabeled.jsonl",
            device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    x = "trigger_emotions"
    print(f"\n\n############################################## {x} ##############################################")
    model = ft_emos.finetune(prompt=f"./prompts/prompt_{x}.yml", target=x, save_in=f"./models/{x}_Qwen35-4B")

    ft_span = finetunning_llm(
            data_path = "./data/HopeEXP_Train.jsonl",
            data_path_test = "./data/HopeEXP_Test_unlabeled.jsonl",
            device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    x = "span_annotations"
    print(f"\n\n############################################## {x} ##############################################")
    model = ft_span.finetune(prompt=f"./prompts/prompt_{x}.yml", target=x, save_in=f"./models/{x}_Qwen35-4B")    


    this_is_the_most_important_function()
    print("\n\n\n HAVE A NICE DAY :)")