<!-- # Project Name

This software project accompanies the research paper, [Paper title](https://arxiv.org).

Brief description of the project.

## Documentation

## Getting Started  -->

# <img src="figs/MarineGPT_logo.png" alt="The logo of MarineGPT" width="40" height="40"> MarineGPT: Unlocking Secrets of "Ocean" to the Public

<a href="https://hkust-vgd.github.io/MarineGPT/"><img src="https://img.shields.io/badge/WEBSITE-Visit%20project%20page-blue?style=for-the-badge"></a>

<a href="https://arxiv.org/pdf/2310.13596.pdf"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>

A first vision-language model specially designed for the marine domain. It could generate more **sensitive**, **informative**, and **scientific** responses as a powerful marine AI assistant.

[Ziqiang Zheng](https://zhengziqiang.github.io/), [Jipeng Zhang](https://2003pro.github.io/), [Tuan-Anh Vu](https://tuananh1007.github.io/), [Shizhe Diao](https://shizhediao.github.io/), [Yue Him Wong Tim](https://scholar.google.com/citations?user=M5j3ZiQAAAAJ&hl=zh-CN), [Sai-Kit Yeung](https://saikit.org/) 

## 📢 News

[Mar.2 2024] We include [LLaVA1.5](https://github.com/haotian-liu/LLaVA) for comparison and embed LLaVA into our MarineGPT. Pre-trained models will be uploaded soon! 

[Feb.26 2024] MarineGPT now supports the [GEMMA](https://blog.google/technology/developers/gemma-open-models/) and we released the pre-trained models of GEMMA.

[Feb.19 2024] We released the pre-trained models of MarineGPT.



## Online Demo

Coming Soon.

## Overview

<p align="center">
    <img src="figs/marinegpt_framework.png" width="100%"></a> <br>
    Framework of MarineGPT.
</p>

Key Contributions:
* MarineGPT - **Domain-specific (marine) MLLM + Instruction-following tuning** enable fine-grained marine object recognition and yield sensitive, informative and scientific response.
* Marine-5M Dataset (~5M) - A **Large-scale, Diverse, Broad-coverage** marine image-text dataset for promoting aligning visual-and-language modalities.
* A marine-specific data generation pipeline to create diverse (image, instruction, output) instruction-following training data.

Potential Applications of MarineGPT:
* Scale up Marine Organism Recognition.
* Monitoring.
* Centralized Platform.
* Interdisciplinary Research.
* General Public Access.



## Abstract
Large language models (LLMs), such as ChatGPT/GPT-4, have proven to be powerful tools in promoting the user experience as an AI assistant. 
The continuous works are proposing multi-modal large language models (MLLM), 
empowering LLMs with the ability to sense multiple modality inputs through constructing a joint semantic space (*e.g.* visual-text space). 
Though significant success was achieved in LLMs and MLLMs, exploring LLMs and MLLMs in domain-specific applications that 
required domain-specific knowledge and expertise has been less conducted, especially for **marine domain**. 
Different from general-purpose MLLMs, the marine-specific MLLM is required to yield much more **sensitive**, **informative**, and **scientific** responses. 
In this work, we demonstrate that the existing MLLMs optimized on huge amounts of readily available general-purpose training data show a minimal ability 
to understand domain-specific intents and then generate informative and satisfactory responses. To address these issues, we propose **MarineGPT**, 
the first vision-language model specially designed for the marine domain, unlocking the secrets of the ocean to the public. 
We present our **Marine-5M** dataset with more than 5 million marine image-text pairs to inject domain-specific marine knowledge into our model 
and achieve better marine vision and language alignment. Our MarineGPT not only pushes the boundaries of marine understanding to the general public 
but also offers a standard protocol for adapting a general-purpose assistant to downstream domain-specific experts. We pave the way for a wide range of marine applications 
while setting valuable data and pre-trained models for future research in both academic and industrial communities.

## Results
* Comparison with [MiniGPT-4](https://minigpt-4.github.io/) and [GPT-4V](https://chat.openai.com/).

<p align="center">
    <img src="figs/comparison.png" width="100%"></a> <br>
</p>

* Recognizing various marine objects.

<p align="center">
    <img src="figs/wide.png" width="100%"></a> <br>
</p>

* Fine-grained marine object recognition.

<p align="center">
    <img src="figs/fine_grained.png" width="100%"></a> <br>
</p>

* Comprehensive multi-round conversation.

<p align="center">
    <img src="figs/comprehensive.png" width="100%"></a> <br>
</p>

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/hkust-vgd/MarineGPT
cd MarineGPT
conda env create -f environment.yml
conda activate marinegpt
```


**2. Prepare the pretrained LLM weights**

**MarineGPT** is based on Vicuna V0 7B/13B. 
Please download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

|                                           Vicuna V0 13B                                           |                                          Vicuna V0 7B                                          |
:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
[Downlad](https://huggingface.co/Vision-CAIR/vicuna/tree/main) | [Download](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) 

Then, set the variable *llama_model* in the model config file to the LLM weight path.

```
### modify the path of LLM weights in Line 16 of marinegpt/configs/models/marinegpt.yaml
llama_model: "/path/to/LLM_weights/"
```

**MarineGPT** can also support GEMMA-2B/7B. 
Please download the corresponding LLM weights from the following huggingface space via clone the repository using git-lfs.

Pre-trained GEMMA models
|                                           GEMMA 2B                                           |                                          GEMMA 7B                                          |
:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
[Downlad](https://huggingface.co/google/gemma-2b) | [Download](https://huggingface.co/google/gemma-7b)

GEMMA models after instruction tuning
|                                           GEMMA 2B-it                                           |                                          GEMMA 7B-it                                          |
:----------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:
[Downlad](https://huggingface.co/google/gemma-2b-it) | [Download](https://huggingface.co/google/gemma-7b-it) 

Then, set the variable *gemma_model* in the model config file to the LLM weight path.

```
### modify the path of LLM weights in Line 16 of marinegpt/configs/models/marinegpt_gemma.yaml
llama_model: "/path/to/GEMMA_weights/"
```

For **MarineGPT**, we will also plan to support the LLaMA and LLaMA 2 version. We will release the trained weights very soon.

**Vicuna**

| MarineGPT Stage 1 (Vicuna 13B)                                                                           | MarineGPT Stage 2 (Vicuna 13B)                                                                 | MarineGPT stage 1 (Vicuna 7B)                                                                  | MarineGPT stage 2 (Vicuna 7B)               |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [Download](https://www.dropbox.com/scl/fi/ot7eefgrnq0jrx51mwktx/marinegpt_vicuna_13B_stage1_ckpt.pth?rlkey=b17nkct52abl5wdomjsvfrf1j&dl=0) | [Download](https://www.dropbox.com/scl/fi/zo19kqd7ay1h7frbxptnw/marinegpt_vicuna_13B_stage2_ckpt.pth?rlkey=2of6jkiaqdu1i44rzvxeg6hlu&dl=0) | [Download](https://www.dropbox.com/scl/fi/lmxwkp96u326h82lssj7w/marinegpt_vicuna_7B_stage1_ckpt.pth?rlkey=rfnup88u9y3go7vs8xowr73n5&dl=0) | [Download](https://www.dropbox.com/scl/fi/8uimfr9vjk8sa6yyvvnbk/marinegpt_vicuna_7B_stage2_ckpt.pth?rlkey=4cwn4cmgi8gjnqfyds2aqnw8s&dl=0) |

**GEMMA**

| MarineGPT Stage 2 (GEMMA 2B)                                                                           | MarineGPT Stage 2 (GENNA 2B-it)                                                                 | MarineGPT stage 2 (GEMMA 7B)                                                                  | MarineGPT stage 2 (GEMMA 7B-it)               |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [Download](https://www.dropbox.com/scl/fi/ggk3gqyjq70szmti62jxd/marinegpt_gemma_2B_stage2_ckpt.pth?rlkey=tzjkil1aqg4ambarl5yyy9amz&dl=0) | [Download](https://www.dropbox.com/scl/fi/otqak3ygc5qe9bdu6hthm/marinegpt_gemma_2B_it_stage2_ckpt.pth?rlkey=ejso60ger89mmplu8nc1o0mui&dl=0) | [Download](https://www.dropbox.com/scl/fi/rs39ofzfpt7fqon748m9x/marinegpt_gemma_7B_stage2_ckpt.pth?rlkey=xkfjslo5msoqt1i41p6uxh8gy&dl=0) | [Download](https://www.dropbox.com/scl/fi/7a0v6upm4ezcbysh3vu2p/marinegpt_gemma_7B_it_stage2_ckpt.pth?rlkey=kmdml4ktho3euajmmt55gqunt&dl=0) |

For **MarineGPT**, set the path to the pretrained checkpoint in the evaluation config file 
in [eval_configs/marinegpt_eval.yaml](eval_configs/marinegpt_eval.yaml#L11) at Line 11.

**3. Launching Demo Locally**

**Vicuna**

For MarineGPT, run

```
python demo.py --cfg-path eval_configs/marinegpt_eval.yaml  --gpu-id 0
```
Please specify the path of pre-trained checkpoints (stage 1 or stage 2; Vicuda 7B or Vicuna 13B) in [eval_configs/marinegpt_eval.yaml](eval_configs/marinegpt_eval.yaml#L11) at Line 11.   

```
### modify the path of pre-trained ckpts in Line 1q of eval_configs/marinegpt_eval.yaml
ckpt: './ckpt/vicuna_7B/stage1/marinegpt_vicuna_7B_stage1_ckpt.pth'
```

**GEMMA**

For MarineGPT, run

```
python demo.py --cfg-path eval_configs/marinegpt_gemma_eval.yaml  --gpu-id 0 --model_type gemma_model
```
Please specify the path of pre-trained checkpoints in [eval_configs/marinegpt_gemma_eval.yaml](eval_configs/marinegpt_gemma_eval.yaml#L11) at Line 11.   

```
### modify the path of pre-trained ckpts in Line 1q of eval_configs/marinegpt_eval.yaml
ckpt: './ckpt/gemma_2B/stage2/marinegpt_gemma_2B_stage2_ckpt.pth'
```

**4. Other applications**

MarineGPT could also support to generate feature embedding and captions for the visual images

```
### generate the feature embedding for retrieval 
python generate_embeddings.py --cfg-path eval_configs/marinegpt_eval.yaml  --gpu-id 0 --img_path ./img_path --output_path ./output_path
```

```
### generate the feature embedding for retrieval 
python generate_captions_for_imgs.py --cfg-path eval_configs/marinegpt_eval.yaml  --gpu-id 0 --img_path ./img_path
```


### Training

**1. Datasets**

We will provide more details of our training data. 

**2. Implementation Details**

Stage 1 (pre-training): please refer to [train_configs/marinegpt_stage1_pretrain.yaml](train_configs/marinegpt_stage1_pretrain.yaml)

Stage 2 (finetuning): please refer to [train_configs/marinegpt_stage2_finetune.yaml](train_configs/marinegpt_stage2_finetune.yaml)

More implementation details will be added soon.

## Acknowledgement

+ [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) The model architecture of MarineGPT follows BLIP-2. Please check this great open-source work if you are not familiar with VLMs!
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Our codes are mainly based on MiniGPT-4. Thanks for their contributions to the whole community.
+ [Lavis](https://github.com/salesforce/LAVIS) Our project is also built upon Lavis!
+ [Vicuna](https://github.com/lm-sys/FastChat) A powerful and open-source LLM to understand the user intents!
+ [LLaVA](https://github.com/haotian-liu/LLaVA) A powerful and open-source MLLM!

##  Citing MarineGPT

If you find MarineGPT helpful, please consider citing:
```
@misc{zheng2023marinegpt,
      title={MarineGPT: Unlocking Secrets of "Ocean" to the Public}, 
      author={Ziqiang Zheng and Jipeng Zhang and Tuan-Anh Vu and Shizhe Diao and Yue Him Wong Tim and Sai-Kit Yeung},
      year={2023},
      eprint={2310.13596},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
