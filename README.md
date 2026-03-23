<h1 align="center">
  <img src="./asset/logo.png" width="40" alt="logo"> HERMES
</h1>
<p align="center">
  <b>KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding</b>
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://hermes-streaming.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2601.14724)
[![HF Paper](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/papers/2601.14724)

</div>

## 🔥 News

- **[2026.03.23]** Full code released!
- **[2025.01.23]** HERMES reached **#3 Paper of the day** on [Hugging Face Daily Papers](https://huggingface.co/papers/2601.14724)!
- **[2025.01.21]** HERMES is available on [arXiv](https://arxiv.org/abs/2601.14724).


## 🛠️ Installation

For **LLaVA** model inference:
```bash
conda create -n hermes-llava python=3.12 -y
conda activate hermes-llava
pip install -r requirements_llava.txt
pip install flash-attn --no-build-isolation
```

For **Qwen2.5-VL** model inference:
```bash
conda create -n hermes-qwen python=3.12 -y
conda activate hermes-qwen
pip install -r requirements_qwen.txt
pip install flash-attn --no-build-isolation
```


## 📦 Preparation

Create a `models` directory and download the model weights from HuggingFace:

```bash
mkdir models
```

We support the following models (choose one or more):

| Model Family | Model | HuggingFace Link |
|:---:|:---:|:---:|
| LLaVA-OneVision | llava-onevision-qwen2-0.5b-ov-hf | [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) |
| LLaVA-OneVision | llava-onevision-qwen2-7b-ov-hf | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) |
| LLaVA-OneVision | llava-onevision-qwen2-72b-ov-hf | [llava-hf/llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf) |
| Qwen2.5-VL | Qwen2.5-VL-3B-Instruct | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL | Qwen2.5-VL-7B-Instruct | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL | Qwen2.5-VL-32B-Instruct | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) |


### Data Preparation

Download the benchmark videos from their official sources and place them according to the paths specified in the annotation files:

**Streaming Benchmarks:**

| Benchmark | Video Path | Official Source |
|:---:|:---:|:---:|
| StreamingBench | `/data/streamingbench/videos/` | 🤗 [StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench) |
| OVO-Bench | `/data/ovobench/videos/` | 🤗 [OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) |
| RVS-Ego | `/data/rvs/ego/videos/` | 🤗 [RVS](https://huggingface.co/datasets/Becomebright/RVS) |
| RVS-Movie | `/data/rvs/movie/videos/` | 🤗 [RVS](https://huggingface.co/datasets/Becomebright/RVS) |

**Offline Benchmarks:**

| Benchmark | Video Path | Official Source |
|:---:|:---:|:---:|
| VideoMME | `/data/videomme/videos/` | 🤗 [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME) |
| MVBench | `/data/mvbench/videos/` | 🤗 [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) |
| EgoSchema | `/data/egoschema/videos/` | 🤗 [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema) |

The annotation JSON files contain the same information as officially provided, with formatting adjustments to adapt to our codebase.

After preparation, the project structure should look like this:

```
HERMES/
├── asset/
│   └── logo.png
├── data/
│   ├── egoschema/
│   │   ├── videos/
│   │   └── egoschema.json
│   ├── mvbench/
│   │   ├── videos/
│   │   └── mvbench.json
│   ├── ovobench/
│   │   ├── videos/
│   │   └── ovobench_realtime_backeward.json
│   ├── rvs/
│   │   ├── ego/
│   │   │   ├── videos/
│   │   │   └── ego4d_oe.json
│   │   └── movie/
│   │       ├── videos/
│   │       └── movienet_oe.json
│   ├── streamingbench/
│   │   ├── videos/
│   │   └── streamingbench_realtime.json
│   └── videomme/
│       ├── videos/
│       └── videomme.json
├── eval/
│   ├── eval_multiple_choice.py
│   └── eval_open_ended.py
├── inference/
│   ├── abstract_hermes.py
│   ├── llavaov_hermes.py
│   ├── qwenvl_hermes.py
│   ├── reindex_1d.py
│   └── reindex_3d.py
├── models/
│   ├── llava-onevision-qwen2-0.5b-ov-hf/
│   ├── llava-onevision-qwen2-7b-ov-hf/
│   ├── llava-onevision-qwen2-72b-ov-hf/
│   ├── Qwen2.5-VL-3B-Instruct/
│   ├── Qwen2.5-VL-7B-Instruct/
│   └── Qwen2.5-VL-32B-Instruct/
├── scripts/
│   └── run_infer.sh
├── video_qa/
│   ├── base.py
│   ├── hermes_vqa.py
│   └── run_infer.py
├── LICENSE
├── README.md
├── requirements_llava.txt
└── requirements_qwen.txt
```


## 🚀 Inference

Simply run the inference script:

```bash
bash scripts/run_infer.sh
```

Here is the content of `scripts/run_infer.sh`:

```bash
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH

num_chunks=8
model=llava_ov_7b
dataset=streamingbench

python video_qa/run_infer.py \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --kv_size 6000
```

**Arguments:**

| Argument | Description |
|:---|:---|
| `--model` | Model to use. Options: `llava_ov_0.5b`, `llava_ov_7b`, `llava_ov_72b`, `qwen2.5_vl_3b`, `qwen2.5_vl_7b`, `qwen2.5_vl_32b` |
| `--dataset` | Benchmark dataset. Options: `videomme`, `mvbench`, `egoschema`, `rvs_ego`, `rvs_movie`, `ovobench`, `streamingbench` |
| `--num_chunks` | Number of parallel processes for evaluation, typically set to the number of GPUs |
| `--sample_fps` | Frame sampling rate (frames per second) from the video |
| `--kv_size` | Maximum KV cache size for HERMES hierarchical memory management |
| `--only_eval` | If set, skip inference and only run evaluation on existing results |


## 📊 Evaluation

The evaluation scripts compute metrics on the inference results:

- **Multiple-choice benchmarks** (VideoMME, MVBench, EgoSchema, OVBench, StreamingBench) are evaluated by `eval/eval_multiple_choice.py`, which takes a subcommand as its first argument:

| Subcommand | Description | Used by |
|:---|:---|:---|
| `general` | Compute overall accuracy, task-specific breakdown (auto-detects OVBench / StreamingBench), and prediction error analysis | MVBench, OVBench, StreamingBench, VideoMME |
| `videomme` | Report accuracy broken down by video duration (short / medium / long) | VideoMME |
| `egoschema` | Generate EgoSchema submission CSV file | EgoSchema |

```bash
python eval/eval_multiple_choice.py general --results_path results/llava_ov_7b/streamingbench/fps0.5-kv6000/results.csv
```

- **Open-ended benchmarks** (RVS-Ego, RVS-Movie) are evaluated by `eval/eval_open_ended.py`, which uses GPT for answer scoring:

```bash
python eval/eval_open_ended.py \
    --pred_path results/llava_ov_7b/rvs_ego/fps0.5-kv6000/results.csv \
    --output_dir results/llava_ov_7b/rvs_ego/fps0.5-kv6000/tmp \
    --output_json results/llava_ov_7b/rvs_ego/fps0.5-kv6000/results.json
```


## 📧 Contact

For any questions regarding the paper or the technical implementation, please feel free to contact haowei.zhang123@gmail.com


## 🙏 Acknowledgements

Our codebase is built upon [ReKV](https://github.com/Becomebright/ReKV). We gratefully acknowledge their contributions to the community.


## 📝 Citation

If you find our work useful for research, please cite our paper and give us a precious star 😄:

```bibtex
@misc{zhang2026hermeskvcachehierarchical,
      title={HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding}, 
      author={Haowei Zhang and Shudong Yang and Jinlan Fu and See-Kiong Ng and Xipeng Qiu},
      year={2026},
      eprint={2601.14724},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14724}, 
}
```
