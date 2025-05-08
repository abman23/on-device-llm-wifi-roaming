# On-Device LLM for Context-Aware Wi-Fi Roaming
This is the official repository of our paper "On-Device LLM for Context-Aware Wi-Fi Roaming".

![CHO demo clip](images/CHO_animation.gif)
## Overview

Context-aware Handover (CHO) is a Large Language Model (LLM)-based framework for improved WLAN handover decisions. By leveraging the strong pattern recognization capability of LLM and the rich contextual information of WLAN, CHO outperforms several traditional rule-based methods in the sense of achieving a balance between handover frequency and signal strength.

## Highlights

- We introduce a novel LLM-based framework for *adaptive handover* based on the contextual information, which strikes a balance between handover frequency and signal strength.
- We include the *inference scripts* for both two tasks in the paper: (1) context-aware AP choice, (2) online roaming optimization.
- We release a *Wi-Fi roaming dataset* collected from real-world environments, which can be used for model training and evaluation in related tasks.

## Preparation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/abman23/llm-handover.git
   ```

2. **Download the dataset**
- Download the dataset from [here](https://drive.google.com/file/d/1U-xHQc8mTHOiScTuVzCZL8ju1XM_3OUU/view?usp=drive_link).
- Extract the ZIP archive to the `data/` directory.

2. **Set up the Virtual Environment**
   ```sh
   conda create -n cho_env python=3.10.14
   conda activate cho_env
   pip install -r requirements.txt
   ```

3. **Deploy LLMs Locally**
- Install and launch [Ollama](https://ollama.com/).
- Create a local LLM in Ollama.
    ```sh
    ollama pull llama3.2:1b-instruct-q2_K
    ```
  You can replace `llama3.2:1b-instruct-q2_K` by any other model hosted by [Ollama](https://ollama.com/search), and then modify the arguments in the following inference scripts.
  
## Usage

### LLM inference
For task (1) context-aware AP choice:
```sh
python inference_task1.py --thr -70 --data hybrid_2.json --model llama3.2:1b-instruct-q2_K
```
For task (2) online roaming optimization:
```sh
python inference_task2.py --thr -60 -70 -80 --data hybrid_2.json --interval 30 --model llama3.2:1b-instruct-q2_K
```

### Test your own model
You can fine-tune your own model and test it out under our CHO setting.
For the local deployment of fine-tuned LLM in Ollama, please refer to this [tutorial](https://github.com/ollama/ollama/blob/main/docs/import.md).

## Citation
```

```
