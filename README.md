# VCR at HopeEXP 2026

[**MSc. Jesús Armenta-Segura**](https://scholar.google.com/citations?user=qFEylJkAAAAJ&hl=en)

This repo contains the code for reproduce my results [at this competition.](https://www.codabench.org/competitions/13563/)

I achieved **1st place** on the multilabel sentiment analysis part, and **2nd place** on all three tasks together (multiclass and multilabel sentiment analysis, and textual span extraction).

[LINK TO MY PAPER](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

### What I did?
Unsloth **finetunning of Qwen 3.5 (4B)**, using a carefully designed custom prompt, for all three tasks.

Boring (ultra boring) tech details:
I used an 8-bit quantization, and I freezed all "hidden layers" but the Queries, Keys and Values matrices, as well as the output layers (in the case of Qwen, a simple perceptron working as a decoder, ultra lol how that thing worked that well? xd).
After finding poor initial results, I desided to also include the gated mechanisms of the models. I ~~stole~~ obtained that idea from [this paper](https://arxiv.org/abs/2507.10996).
However, I personally suspect that tunning gated mechanisms worked as a pseudo-dropout mechanism? (token replication told me that the model was memorizing prompts, however learning curves behaved well so I did what not too many ppl does: read papers xdxdxdx).


## INSTRUCTIONS FOR REPRODUCIBILITY:

1. Get the data. (Contact the organizers).
2. Initialize a virtual enviroment and install all packages at the requirements.txt version.
3. Run main.py, you will need private keys on a params.json file.
About the params.json file:
'''
{
    "data_URL": "https://www.codabench.org/contact/the/organizers/for/the/URL/to/the/database/:)",
    "llm_name": "unsloth/Qwen3.5-4B",
    "lora_modules": ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj","lm_head"],
}

'''
I guess, if something is missing just fix it idk...
4. Enjoy.
