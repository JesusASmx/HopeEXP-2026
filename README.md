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
My initial approach was **finetunning Qwen 3.5 (4B)**, using a carefully designed custom prompt, for all three tasks. However, although results on 2nd, 3rd, 4th and 5th task were good, the first one was very standard (a simple classification task) and 2019-model RoBERTa brought the best results. So I stuck on that. MOREOVER, multitasking was a blatant disaster: I whitnessed totally amused how Qwen 3.5 (4B) was unnable to surpass the 0.4 score thresshold on each task (we are still far from AGI).

#### Tech details:
**For RoBERTa:**
-I used the XML version of the model, which supports multilanguage.
-No quantization was required.
-Batch size was 16.
-Other hyperparameters are referred on the paper, along with the learning curves.

**For Qwen:**
-I used the Unsloth implementation of the model, 4B version.
-I used 4-bit quantization.
-LowRank was also obviously involved. I freezed all "hidden layers" but the Queries, Keys and Values matrices, as well as the output layers (in the case of Qwen, a simple perceptron working as a decoder).
-After poor initial results, I desided to also include the gated mechanisms of the models. I ~~stole~~ obtained that idea from [this paper](https://arxiv.org/abs/2507.10996). However, I personally suspect that tunning gated mechanisms worked as a pseudo-dropout mechanism? (token replication told me that the model was memorizing prompts. However, learning curves behaved well so...).

***

## INSTRUCTIONS FOR REPRODUCIBILITY:

1. Get the data. (Contact the organizers).
2. Initialize a virtual enviroment and install all packages at the requirements.txt version.
3. Run main.py, you will need private keys on a params.json file. This repo includes a template.
4. Enjoy having your own version of the winning system.

   
CAVEAT: Due hard disk constraints, I didn't saved the models locally and I forgot to include automatically savers.

***

<div align="center">
  <img width="311" height="210" alt="image" src="https://github.com/user-attachments/assets/303e18e7-0682-4554-8759-cad8a4426b64" />
  Bro is angry cus he won't be 1st author.
</div>



