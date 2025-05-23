# On-Device LLM for Context-Aware Wi-Fi Roaming
[![arXiv](https://img.shields.io/badge/arXiv-2505.04174-b31b1b.svg)](https://arxiv.org/abs/2505.04174)

This is the official repository for the paper **“On-Device LLM for Context-Aware Wi-Fi Roaming”** (arXiv:2505.04174, 2025).

## Demo Videos
<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=U9QFzw7fJMQ">
        <img src="https://img.youtube.com/vi/U9QFzw7fJMQ/0.jpg" width="280"><br>
        Indoor Campus Demo
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=UR13hyJkE8k">
        <img src="https://img.youtube.com/vi/UR13hyJkE8k/0.jpg" width="280"><br>
        Outdoor Street Demo
      </a>
    </td>
  </tr>
</table>

![CHO demo clip](images/CHO_animation.gif)

## Overview

We propose a novel cross-layer framework using an on-device Large Language Model (LLM) for context-aware Wi-Fi roaming. This approach integrates high-level reasoning in the application layer with real-time action at the PHY/MAC layers. Specifically, our LLM handles two critical roaming tasks: (1) context-driven AP selection leveraging environmental context (e.g., location, time), and (2) adaptive roaming threshold adjustments. Optimized for practical edge deployment—demonstrated effectively on consumer-grade hardware like a MacBook Pro—via quantization and efficient fine-tuning, our framework significantly enhances roaming decisions, balancing handover stability and signal quality.

## Highlights

- **Cross-layer control**: First demonstration of an on-device LLM performing real-time PHY/MAC layer actions based on application-layer context reasoning.
- **Adaptive and efficient**: Utilizes prompt engineering, parameter-efficient fine-tuning (LoRA), and quantization to achieve fast inference suitable for edge hardware, including consumer devices like MacBook Pro.
- **Real-world validation**: Evaluated comprehensively with indoor and outdoor Wi-Fi datasets, outperforming conventional heuristic and DRL methods.

## Quick Start

### 1. Clone
   ```sh
   git clone https://github.com/abman23/on-device-llm-wifi-roaming.git
   ```

### 2. Dataset
1. Download from https://drive.google.com/file/d/1U-xHQc8mTHOiScTuVzCZL8ju1XM_3OUU
2. Unzip into data/

### 3. Environment**
   ```sh
   conda create -n cho_env python=3.10.14
   conda activate cho_env
   pip install -r requirements.txt
   ```

### 4. **Local LLM**
1. Install and launch [Ollama](https://ollama.com/).
- Create a local LLM in Ollama.
    ```sh
    ollama pull llama3.2:1b-instruct-q2_K
    ```
  You can replace `llama3.2:1b-instruct-q2_K` by any other model hosted by [Ollama](https://ollama.com/search), and then modify the arguments in the following inference scripts.
  
### 5. Usage

### LLM inference
For task (1) context-aware AP choice:
   ```sh
   python inference_task1.py --thr -70 --data hybrid_2.json --model llama3.2:1b-instruct-q2_K
   ```
For task (2) online roaming optimization:
   ```sh
   python inference_task2.py --thr -60 -70 -80 --data hybrid_2.json --interval 30 --model llama3.2:1b-instruct-q2_K
   ```

### Fine-Tune & Evaluate Your Own Model
You can fine-tune your own model and test it out under our CHO setting.
For the local deployment of fine-tuned LLM in Ollama, please refer to this [tutorial](https://github.com/ollama/ollama/blob/main/docs/import.md).

## Citation
```
@article{lee2025device,
  title   = {On-Device LLM for Context-Aware Wi-Fi Roaming},
  author  = {Lee, Ju-Hyung and Lu, Yanqing},
  journal = {arXiv preprint arXiv:2505.04174},
  year    = {2025}
}
```
