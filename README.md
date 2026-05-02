# VCR at HopeEXP 2026

[**MSc. Jesús Armenta-Segura**](https://scholar.google.com/citations?user=qFEylJkAAAAJ&hl=en)

This repo contains the code for reproduce my results [at this competition.](https://www.codabench.org/competitions/13563/)

```
🥇🥇 I am pleased to announce that I obtained the 1st PLACE on this competition!! 🥇🥇
```

<div align="center">
  I want to thank PhD. Chipokles for this award.
  <img width="200" height="190" alt="image" src="https://github.com/user-attachments/assets/0ad5770e-0bee-48ea-9aa8-55c0eec09fc4" />
  He did nothing but c'mon isn't he cute af (accurately fluffy) ??
</div>

[LINK TO MY PAPER (comming soon)](https://scholar.google.com/citations?user=qFEylJkAAAAJ&hl=en)

***

### What I did?
My initial approach was **finetunning Qwen 3.5 (4B)**, using a carefully designed custom prompt, for all three tasks. However, although results on 2nd, 3rd, 4th and 5th task were good (one different instance for each task), the first task was a very standard classification task, and a 2019-model RoBERTa brought by far the best results.

MOREOVER, multitasking, an approach proven successful on previous shared tasks, was a blatant disaster: I whitnessed amused how Qwen 3.5 (4B) shattered all the previous conceptions about multitasking on traditional models and small deep neural networks, and was unnable to surpass the 0.4 score thresshold on each task (we are still far from AGI).

#### Tech details:
**For RoBERTa:**
- I used the XML version of the model, which supports multilanguage.
- No quantization was required.
- Batch size was 16.
- Other hyperparameters are referred on the paper, along with the learning curves.

**For Qwen:**
- I used the Unsloth implementation of the model, 4B version.
- I used 4-bit quantization.
- Lets play a game! Guess the Batch size! HINTS: A) Only 12g of VRAM, B) less than 3, more than 1, C) Not 16.
- LowRank was also obviously involved. I freezed all "hidden layers" but the Queries, Keys and Values matrices, as well as the mere output layers (in the case of Qwen, a simple perceptron collocated after the MoE-based multiheadattention-decoder).
- After poor initial results, I decided to also include the gated mechanisms of the models. I ~~stole~~ obtained that idea from [this paper](https://arxiv.org/abs/2507.10996). The enhancement was dramatic. I personally suspect that tunning them worked as a pseudo-dropout mechanism... token replication told me that the model was memorizing prompts. However, learning curves behaved well so... simulated dropout mechanisms will be for now *(I am boneless. However, the other day I broke two bones, hence I have at least two bones. Strong evidence suggest the existence of a third one)*.

***

## INSTRUCTIONS FOR REPRODUCIBILITY:

1. Get the data. (Contact the organizers).
2. Initialize a virtual enviroment and install all packages at the requirements.txt version.
3. Run main.py, you will need private keys on a params.json file. This repo includes a template.
4. Enjoy having your own version of the winning system.

   
**CAVEAT**: Due hard disk constraints, I didn't saved the models locally and I forgot to include automatic savers on the code. They can be manually added, very easily.

**DECLARATION**: This repo was not vibecoded. No generative language model was used to write any snipet of it. However, considerable passages of it are adaptations from the follow tutorials:
- Unsloth tutorial: https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Huggingface tutorial: https://huggingface.co/learn/llm-course/chapter3/3
- The starting kit provided by the organizers of this shared task (very useful, thank you!)

***

<div align="center">
  <img width="340" height="210" alt="image" src="https://github.com/user-attachments/assets/303e18e7-0682-4554-8759-cad8a4426b64" />
  Bro is angry cus he won't be 1st author.
</div>



