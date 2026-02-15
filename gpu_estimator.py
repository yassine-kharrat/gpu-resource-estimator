#!/usr/bin/env python3
"""
GPU Resource Estimator v2
Estimate VRAM, training time, FLOPs, and costs for AI model training & inference.
Supports 30+ GPUs, 50+ model presets, and 17 task types.

Usage:
    python gpu_estimator.py                     # Interactive mode
    python gpu_estimator.py --model "LLaMA 3 8B" --task "Full Fine-tuning" --gpu "NVIDIA A100 (80GB)"
    python gpu_estimator.py --export results.json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Optional

# ─── Try importing rich/questionary, fallback to plain mode ───────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

# ═══════════════════════════════════════════════════════════════════
# GPU DATABASE
# ═══════════════════════════════════════════════════════════════════
GPU_DATABASE = {
    # ─── NVIDIA Consumer / Entry ───
    "NVIDIA RTX 3070 (8GB)": {"vram": 8, "tflops_fp16": 81, "tflops_fp32": 20.3, "tflops_tf32": 40, "bandwidth": 448, "tdp": 220, "price_hr": 0.25, "nvlink": False, "arch": "Ampere"},
    "NVIDIA RTX 3080 (10GB)": {"vram": 10, "tflops_fp16": 119, "tflops_fp32": 29.8, "tflops_tf32": 59, "bandwidth": 760, "tdp": 320, "price_hr": 0.35, "nvlink": False, "arch": "Ampere"},
    "NVIDIA T4 (16GB)": {"vram": 16, "tflops_fp16": 65, "tflops_fp32": 8.1, "tflops_tf32": 0, "bandwidth": 300, "tdp": 70, "price_hr": 0.4, "nvlink": False, "arch": "Turing"},
    "NVIDIA RTX 4070 Ti (12GB)": {"vram": 12, "tflops_fp16": 186, "tflops_fp32": 40, "tflops_tf32": 93, "bandwidth": 504, "tdp": 285, "price_hr": 0.40, "nvlink": False, "arch": "Ada"},
    "NVIDIA RTX 4080 (16GB)": {"vram": 16, "tflops_fp16": 194, "tflops_fp32": 48.7, "tflops_tf32": 97, "bandwidth": 717, "tdp": 320, "price_hr": 0.55, "nvlink": False, "arch": "Ada"},
    "NVIDIA V100 (16GB)": {"vram": 16, "tflops_fp16": 125, "tflops_fp32": 15.7, "tflops_tf32": 0, "bandwidth": 900, "tdp": 300, "price_hr": 1.0, "nvlink": True, "arch": "Volta"},
    # ─── NVIDIA Datacenter Mid-range ───
    "NVIDIA L4 (24GB)": {"vram": 24, "tflops_fp16": 121, "tflops_fp32": 30.3, "tflops_tf32": 60, "bandwidth": 300, "tdp": 72, "price_hr": 0.60, "nvlink": False, "arch": "Ada"},
    "NVIDIA A10G (24GB)": {"vram": 24, "tflops_fp16": 125, "tflops_fp32": 31.2, "tflops_tf32": 62.5, "bandwidth": 600, "tdp": 300, "price_hr": 0.70, "nvlink": False, "arch": "Ampere"},
    "NVIDIA A10 (24GB)": {"vram": 24, "tflops_fp16": 125, "tflops_fp32": 31.2, "tflops_tf32": 62.5, "bandwidth": 600, "tdp": 150, "price_hr": 0.75, "nvlink": False, "arch": "Ampere"},
    "NVIDIA RTX 4090 (24GB)": {"vram": 24, "tflops_fp16": 330, "tflops_fp32": 82.6, "tflops_tf32": 165, "bandwidth": 1008, "tdp": 450, "price_hr": 0.8, "nvlink": False, "arch": "Ada"},
    "NVIDIA RTX 3090 (24GB)": {"vram": 24, "tflops_fp16": 142, "tflops_fp32": 35.6, "tflops_tf32": 71, "bandwidth": 936, "tdp": 350, "price_hr": 0.5, "nvlink": True, "arch": "Ampere"},
    "NVIDIA A30 (24GB)": {"vram": 24, "tflops_fp16": 165, "tflops_fp32": 10.3, "tflops_tf32": 82, "bandwidth": 933, "tdp": 165, "price_hr": 1.10, "nvlink": True, "arch": "Ampere"},
    "NVIDIA V100 (32GB)": {"vram": 32, "tflops_fp16": 125, "tflops_fp32": 15.7, "tflops_tf32": 0, "bandwidth": 900, "tdp": 300, "price_hr": 1.30, "nvlink": True, "arch": "Volta"},
    # ─── NVIDIA Datacenter High-end ───
    "NVIDIA A100 (40GB)": {"vram": 40, "tflops_fp16": 312, "tflops_fp32": 19.5, "tflops_tf32": 156, "bandwidth": 1555, "tdp": 300, "price_hr": 2.2, "nvlink": True, "arch": "Ampere"},
    "NVIDIA A40 (48GB)": {"vram": 48, "tflops_fp16": 150, "tflops_fp32": 37.4, "tflops_tf32": 75, "bandwidth": 696, "tdp": 300, "price_hr": 1.50, "nvlink": True, "arch": "Ampere"},
    "NVIDIA RTX A6000 (48GB)": {"vram": 48, "tflops_fp16": 155, "tflops_fp32": 38.7, "tflops_tf32": 77, "bandwidth": 768, "tdp": 300, "price_hr": 1.40, "nvlink": True, "arch": "Ampere"},
    "NVIDIA L40 (48GB)": {"vram": 48, "tflops_fp16": 181, "tflops_fp32": 90.5, "tflops_tf32": 181, "bandwidth": 864, "tdp": 300, "price_hr": 1.60, "nvlink": False, "arch": "Ada"},
    "NVIDIA L40S (48GB)": {"vram": 48, "tflops_fp16": 362, "tflops_fp32": 91, "tflops_tf32": 183, "bandwidth": 864, "tdp": 350, "price_hr": 1.8, "nvlink": False, "arch": "Ada"},
    "NVIDIA A100 (80GB)": {"vram": 80, "tflops_fp16": 312, "tflops_fp32": 19.5, "tflops_tf32": 156, "bandwidth": 2039, "tdp": 300, "price_hr": 3.5, "nvlink": True, "arch": "Ampere"},
    "NVIDIA H100 (80GB)": {"vram": 80, "tflops_fp16": 990, "tflops_fp32": 51, "tflops_tf32": 495, "bandwidth": 3350, "tdp": 700, "price_hr": 5.5, "nvlink": True, "arch": "Hopper"},
    "NVIDIA H100 NVL (94GB)": {"vram": 94, "tflops_fp16": 1979, "tflops_fp32": 67, "tflops_tf32": 989, "bandwidth": 3958, "tdp": 400, "price_hr": 6.00, "nvlink": True, "arch": "Hopper"},
    "NVIDIA H200 (141GB)": {"vram": 141, "tflops_fp16": 990, "tflops_fp32": 51, "tflops_tf32": 495, "bandwidth": 4800, "tdp": 700, "price_hr": 7.0, "nvlink": True, "arch": "Hopper"},
    # ─── NVIDIA Blackwell ───
    "NVIDIA B100 (192GB)": {"vram": 192, "tflops_fp16": 3500, "tflops_fp32": 60, "tflops_tf32": 1750, "bandwidth": 8000, "tdp": 700, "price_hr": 9.00, "nvlink": True, "arch": "Blackwell"},
    "NVIDIA B200 (192GB)": {"vram": 192, "tflops_fp16": 4500, "tflops_fp32": 70, "tflops_tf32": 2250, "bandwidth": 8000, "tdp": 1000, "price_hr": 12.00, "nvlink": True, "arch": "Blackwell"},
    "NVIDIA GB200 (384GB)": {"vram": 384, "tflops_fp16": 9000, "tflops_fp32": 90, "tflops_tf32": 4500, "bandwidth": 16000, "tdp": 1200, "price_hr": 18.00, "nvlink": True, "arch": "Blackwell"},
    # ─── AMD ───
    "AMD MI210 (64GB)": {"vram": 64, "tflops_fp16": 181, "tflops_fp32": 22.6, "tflops_tf32": 90, "bandwidth": 1639, "tdp": 300, "price_hr": 2.00, "nvlink": False, "arch": "CDNA2"},
    "AMD MI250X (128GB)": {"vram": 128, "tflops_fp16": 383, "tflops_fp32": 47.9, "tflops_tf32": 191, "bandwidth": 3277, "tdp": 560, "price_hr": 3.50, "nvlink": False, "arch": "CDNA2"},
    "AMD MI300X (192GB)": {"vram": 192, "tflops_fp16": 1307, "tflops_fp32": 163, "tflops_tf32": 653, "bandwidth": 5300, "tdp": 750, "price_hr": 6.5, "nvlink": False, "arch": "CDNA3"},
    # ─── Google / Intel ───
    "Google TPU v5e (16GB)": {"vram": 16, "tflops_fp16": 197, "tflops_fp32": 98, "tflops_tf32": 0, "bandwidth": 819, "tdp": 200, "price_hr": 1.20, "nvlink": False, "arch": "TPU"},
    "Intel Gaudi 2 (96GB)": {"vram": 96, "tflops_fp16": 420, "tflops_fp32": 52, "tflops_tf32": 210, "bandwidth": 2460, "tdp": 600, "price_hr": 3.00, "nvlink": False, "arch": "Gaudi"},
}

# ═══════════════════════════════════════════════════════════════════
# TASK TYPES
# ═══════════════════════════════════════════════════════════════════
TASK_TYPES = {
    # ─── Training tasks ───
    "Full Fine-tuning": {"multiplier": 18, "description": "Tous les parametres mis a jour", "flops_multiplier": 6, "is_training": True},
    "LoRA / QLoRA": {"multiplier": 1.5, "description": "Adaptation low-rank, ~1-5% params", "flops_multiplier": 6, "is_training": True, "trainable_ratio": 0.03},
    "Pretraining": {"multiplier": 22, "description": "Entrainement from scratch", "flops_multiplier": 6, "is_training": True},
    "Continued Pretraining": {"multiplier": 18, "description": "Pretraining continu sur un domaine specifique", "flops_multiplier": 6, "is_training": True},
    "RLHF / DPO": {"multiplier": 24, "description": "Reward model + policy training", "flops_multiplier": 12, "is_training": True},
    "Distillation (Teacher->Student)": {"multiplier": 14, "description": "Transfert de connaissances d'un modele enseignant", "flops_multiplier": 8, "is_training": True},
    "Embedding / Retrieval": {"multiplier": 4, "description": "Entrainer un modele d'embeddings (Sentence-BERT, E5)", "flops_multiplier": 3, "is_training": True},
    "Classification Fine-tuning": {"multiplier": 10, "description": "Fine-tune pour classification de texte", "flops_multiplier": 5, "is_training": True},
    "Seq2Seq Fine-tuning": {"multiplier": 16, "description": "Fine-tune modele encoder-decoder (T5, BART)", "flops_multiplier": 6, "is_training": True},
    "Vision Fine-tuning": {"multiplier": 16, "description": "Fine-tune modele vision (ViT, CLIP)", "flops_multiplier": 6, "is_training": True},
    "Multimodal (VLM)": {"multiplier": 20, "description": "Entrainement vision-langage", "flops_multiplier": 8, "is_training": True},
    # ─── Inference tasks ───
    "Inference (FP16)": {"multiplier": 2, "description": "Modele charge en FP16", "flops_multiplier": 2, "is_training": False},
    "Inference (FP32)": {"multiplier": 4, "description": "Inference en pleine precision", "flops_multiplier": 2, "is_training": False},
    "Inference (INT8)": {"multiplier": 1.2, "description": "Inference quantifiee 8-bit", "flops_multiplier": 2, "is_training": False},
    "Inference (INT4/GPTQ)": {"multiplier": 0.7, "description": "Inference quantifiee 4-bit GPTQ", "flops_multiplier": 2, "is_training": False},
    "Inference (AWQ)": {"multiplier": 0.6, "description": "Inference 4-bit AWQ", "flops_multiplier": 2, "is_training": False},
    "Inference Batch (FP16)": {"multiplier": 3, "description": "Inference a haut debit (large batch)", "flops_multiplier": 2, "is_training": False},
}

# ═══════════════════════════════════════════════════════════════════
# DEEPSPEED STAGES
# ═══════════════════════════════════════════════════════════════════
DEEPSPEED_STAGES = {
    "None": {"factor": 1, "description": "No ZeRO parallelism", "mem_divisor": 1},
    "ZeRO Stage 1": {"factor": 0.95, "description": "Optimizer state partitioning", "mem_divisor": 4},
    "ZeRO Stage 2": {"factor": 0.90, "description": "+ gradient partitioning", "mem_divisor": 8},
    "ZeRO Stage 3": {"factor": 0.85, "description": "+ parameter partitioning", "mem_divisor": 16},
    "ZeRO-Offload": {"factor": 0.80, "description": "CPU offload (slower)", "mem_divisor": 32},
    "ZeRO-Infinity": {"factor": 0.75, "description": "NVMe offload (slowest)", "mem_divisor": 64},
}

# ═══════════════════════════════════════════════════════════════════
# MODEL PRESETS
# ═══════════════════════════════════════════════════════════════════
MODEL_PRESETS = {
    "Custom": {"params": 0, "layers": 1, "hidden": 512},
    # ─── Tiny / Small Models ───
    "T5 Small (60M)": {"params": 0.06, "layers": 6, "hidden": 512},
    "BERT Base (110M)": {"params": 0.11, "layers": 12, "hidden": 768},
    "T5 Base (220M)": {"params": 0.22, "layers": 12, "hidden": 768},
    "ViT-Large (307M)": {"params": 0.307, "layers": 24, "hidden": 1024},
    "BERT Large (340M)": {"params": 0.34, "layers": 24, "hidden": 1024},
    "CLIP ViT-L/14": {"params": 0.43, "layers": 24, "hidden": 1024},
    "Qwen2.5 0.5B": {"params": 0.5, "layers": 24, "hidden": 896},
    "T5 Large (770M)": {"params": 0.77, "layers": 24, "hidden": 1024},
    # ─── 1B-3B Models ───
    "LLaMA 3.2 1B": {"params": 1.2, "layers": 16, "hidden": 2048},
    "Whisper Large v3": {"params": 1.55, "layers": 32, "hidden": 1280},
    "Gemini Nano (1.8B)": {"params": 1.8, "layers": 18, "hidden": 2048},
    "Gemma 2 2B": {"params": 2.5, "layers": 18, "hidden": 2304},
    "Qwen2.5 3B": {"params": 3, "layers": 36, "hidden": 2048},
    "T5 XL (3B)": {"params": 3, "layers": 24, "hidden": 2048},
    "LLaMA 3.2 3B": {"params": 3.2, "layers": 28, "hidden": 3200},
    "Stable Diffusion XL": {"params": 3.5, "layers": 24, "hidden": 2048},
    "Phi-3 Mini (3.8B)": {"params": 3.8, "layers": 32, "hidden": 3072},
    # ─── 6B-8B Models ───
    "GPT-J 6B": {"params": 6, "layers": 28, "hidden": 4096},
    "Mistral 7B": {"params": 7, "layers": 32, "hidden": 4096},
    "Qwen2.5 7B": {"params": 7, "layers": 28, "hidden": 3584},
    "Phi-3 Small (7B)": {"params": 7, "layers": 32, "hidden": 4096},
    "Falcon 7B": {"params": 7, "layers": 32, "hidden": 4544},
    "LLaMA 3 8B": {"params": 8, "layers": 32, "hidden": 4096},
    "Stable Diffusion 3 (8B)": {"params": 8, "layers": 38, "hidden": 4096},
    # ─── 9B-14B Models ───
    "Gemma 2 9B": {"params": 9.2, "layers": 42, "hidden": 3584},
    "T5 XXL (11B)": {"params": 11, "layers": 24, "hidden": 4096},
    "Qwen2.5 14B": {"params": 14, "layers": 48, "hidden": 5120},
    "Phi-3 Medium (14B)": {"params": 14, "layers": 40, "hidden": 5120},
    "Phi-4 (14B)": {"params": 14, "layers": 40, "hidden": 5120},
    # ─── 24B-40B Models ───
    "Mistral Small 24B": {"params": 24, "layers": 56, "hidden": 5120},
    "Gemma 2 27B": {"params": 27, "layers": 46, "hidden": 4608},
    "Qwen2.5 32B": {"params": 32, "layers": 64, "hidden": 5120},
    "Command-R (35B)": {"params": 35, "layers": 40, "hidden": 8192},
    "Falcon 40B": {"params": 40, "layers": 60, "hidden": 8192},
    # ─── MoE Models ───
    "Mixtral 8x7B": {"params": 47, "layers": 32, "hidden": 4096},
    "Mixtral 8x22B": {"params": 141, "layers": 56, "hidden": 6144},
    # ─── 66B-72B Models ───
    "OPT 66B": {"params": 66, "layers": 64, "hidden": 9216},
    "LLaMA 3 70B": {"params": 70, "layers": 80, "hidden": 8192},
    "LLaMA 3.3 70B": {"params": 70, "layers": 80, "hidden": 8192},
    "Qwen2.5 72B": {"params": 72, "layers": 80, "hidden": 8192},
    # ─── 100B+ Models ───
    "Command-R+ (104B)": {"params": 104, "layers": 64, "hidden": 12288},
    "Mistral Large 123B": {"params": 123, "layers": 88, "hidden": 12288},
    "BLOOM 176B": {"params": 176, "layers": 70, "hidden": 14336},
    "Falcon 180B": {"params": 180, "layers": 80, "hidden": 14848},
    "LLaMA 3.1 405B": {"params": 405, "layers": 126, "hidden": 16384},
    "DeepSeek-V3 671B": {"params": 671, "layers": 128, "hidden": 16384},
    "DeepSeek-R1 671B": {"params": 671, "layers": 128, "hidden": 16384},
}

FRAMEWORKS = {
    "PyTorch + DeepSpeed": 0.45,
    "PyTorch + FSDP": 0.42,
    "PyTorch (vanilla)": 0.35,
    "TensorFlow + XLA": 0.38,
    "JAX + pjit": 0.48,
    "Megatron-LM": 0.52,
}


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def format_bytes(gb: float) -> str:
    if gb >= 1024:
        return f"{gb / 1024:.1f} TB"
    return f"{gb:.1f} GB"


def format_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    if seconds < 86400 * 30:
        return f"{seconds / 86400:.1f} days"
    if seconds < 86400 * 365:
        return f"{seconds / (86400 * 30):.1f} months"
    return f"{seconds / (86400 * 365):.1f} years"


def format_flops(flops: float) -> str:
    if flops >= 1e21:
        return f"{flops / 1e21:.2f} ZetaFLOPs"
    if flops >= 1e18:
        return f"{flops / 1e18:.2f} ExaFLOPs"
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PetaFLOPs"
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TeraFLOPs"
    return f"{flops / 1e9:.2f} GigaFLOPs"


def format_cost(cost: float) -> str:
    if cost >= 1_000_000:
        return f"${cost / 1_000_000:.1f}M"
    if cost >= 1000:
        return f"${cost / 1000:.1f}k"
    return f"${cost:.0f}"


# ═══════════════════════════════════════════════════════════════════
# CORE ESTIMATION ENGINE
# ═══════════════════════════════════════════════════════════════════
@dataclass
class EstimationResult:
    model_size_gb: float = 0.0
    total_vram: float = 0.0
    total_flops: float = 0.0
    flops_per_step: float = 0.0
    total_steps: int = 0
    tokens_per_step: int = 0
    total_tokens: float = 0.0
    trainable_params: float = 0.0
    flops_per_token: float = 0.0
    attention_flops_per_token: float = 0.0
    num_layers: int = 0
    hidden_size: int = 0
    # Memory breakdown
    weights_gb: float = 0.0
    gradients_gb: float = 0.0
    optimizer_gb: float = 0.0
    activations_gb: float = 0.0
    kv_cache_gb: float = 0.0


@dataclass
class TrainingTimeResult:
    training_time_seconds: float = 0.0
    time_per_step: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    effective_mfu: float = 0.0
    scaling_efficiency: float = 1.0
    peak_tflops: float = 0.0
    effective_tflops_total: float = 0.0


def estimate_resources(
    model_params: float,
    task_type: str,
    batch_size: int,
    seq_length: int,
    precision: str,
    grad_checkpoint: bool,
    deepspeed_stage: str,
    num_gpus: int,
    dataset_tokens: float,
    epochs: int,
    num_layers: int,
    hidden_size: int,
) -> EstimationResult:
    bytes_per_param = {"FP32": 4, "FP16/BF16": 2, "INT8": 1, "INT4": 0.5}.get(precision, 2)
    model_size_gb = (model_params * 1e9 * bytes_per_param) / (1024**3)

    task = TASK_TYPES[task_type]
    ds = DEEPSPEED_STAGES[deepspeed_stage]

    weights_gb = model_size_gb
    gradients_gb = 0.0
    optimizer_gb = 0.0
    activations_gb = 0.0
    kv_cache_gb = 0.0

    if not task["is_training"]:
        total_vram = model_size_gb * task["multiplier"]
        kv_cache_gb = (2 * model_params * 1e9 * 2 * seq_length * batch_size) / (1024**3) * 0.001
        total_vram += kv_cache_gb
    else:
        is_lora = task_type == "LoRA / QLoRA"
        gradients_gb = model_size_gb * 0.03 if is_lora else model_size_gb
        optimizer_gb = model_size_gb * 0.1 if is_lora else model_size_gb * 4
        activations_gb = (batch_size * seq_length * model_params * 1e9 * 2) / (1024**3) * 0.00005
        checkpoint_factor = 0.3 if grad_checkpoint else 1.0
        activations_gb *= checkpoint_factor

        raw_vram = model_size_gb + gradients_gb + optimizer_gb + activations_gb

        if deepspeed_stage != "None" and num_gpus > 1:
            partitioned = (gradients_gb + optimizer_gb) / min(num_gpus, ds["mem_divisor"])
            total_vram = model_size_gb + partitioned + activations_gb
            if deepspeed_stage == "ZeRO Stage 3":
                total_vram = (model_size_gb / min(num_gpus, ds["mem_divisor"])) + partitioned + activations_gb
        else:
            total_vram = raw_vram

    # ─── FLOP estimation (Kaplan/Chinchilla) ───
    total_tokens = dataset_tokens * epochs
    P = model_params * 1e9
    trainable_ratio = task.get("trainable_ratio")
    trainable_params = P * trainable_ratio if trainable_ratio else P

    flops_per_token = task["flops_multiplier"] * trainable_params

    L = num_layers if num_layers else max(1, round(math.sqrt(model_params)))
    H = hidden_size if hidden_size else max(512, round(math.sqrt(P / L)))
    attention_flops_per_token = 12 * L * H * seq_length

    total_flops_per_token = flops_per_token + attention_flops_per_token
    total_flops = total_flops_per_token * total_tokens

    tokens_per_step = batch_size * seq_length
    total_steps = math.ceil(total_tokens / tokens_per_step) if tokens_per_step > 0 else 0
    flops_per_step = total_flops_per_token * tokens_per_step

    return EstimationResult(
        model_size_gb=model_size_gb,
        total_vram=total_vram,
        total_flops=total_flops,
        flops_per_step=flops_per_step,
        total_steps=total_steps,
        tokens_per_step=tokens_per_step,
        total_tokens=total_tokens,
        trainable_params=trainable_params,
        flops_per_token=total_flops_per_token,
        attention_flops_per_token=attention_flops_per_token,
        num_layers=L,
        hidden_size=H,
        weights_gb=weights_gb,
        gradients_gb=gradients_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        kv_cache_gb=kv_cache_gb,
    )


def estimate_training_time(
    estimation: EstimationResult,
    gpu_spec: dict,
    num_gpus: int,
    precision: str,
    deepspeed_stage: str,
    framework: str,
) -> TrainingTimeResult:
    peak_tflops = gpu_spec["tflops_fp32"] if precision == "FP32" else gpu_spec["tflops_fp16"]
    if precision == "FP16/BF16" and gpu_spec["tflops_tf32"] > 0:
        peak_tflops = gpu_spec["tflops_fp16"]

    base_mfu = FRAMEWORKS.get(framework, 0.40)

    # GPU architecture bonus
    if gpu_spec["arch"] == "Hopper":
        base_mfu *= 1.1
    elif gpu_spec["arch"] == "Blackwell":
        base_mfu *= 1.15
    elif gpu_spec["arch"] == "CDNA3":
        base_mfu *= 1.05

    # Multi-GPU scaling efficiency
    scaling_efficiency = 1.0
    if num_gpus > 1:
        if gpu_spec["nvlink"]:
            scaling_efficiency = 0.95 ** math.log2(num_gpus)
        else:
            scaling_efficiency = 0.88 ** math.log2(num_gpus)

    ds_overhead = DEEPSPEED_STAGES[deepspeed_stage]["factor"]
    effective_mfu = min(base_mfu * ds_overhead, 0.60)
    effective_tflops = peak_tflops * num_gpus * scaling_efficiency * effective_mfu * 1e12

    if effective_tflops <= 0:
        return TrainingTimeResult()

    training_time_seconds = estimation.total_flops / effective_tflops
    time_per_step = estimation.flops_per_step / effective_tflops if effective_tflops > 0 else 0
    throughput = estimation.tokens_per_step / time_per_step if time_per_step > 0 else 0

    return TrainingTimeResult(
        training_time_seconds=training_time_seconds,
        time_per_step=time_per_step,
        throughput_tokens_per_sec=throughput,
        effective_mfu=effective_mfu,
        scaling_efficiency=scaling_efficiency,
        peak_tflops=peak_tflops,
        effective_tflops_total=effective_tflops / 1e12,
    )


# ═══════════════════════════════════════════════════════════════════
# CODE GENERATORS
# ═══════════════════════════════════════════════════════════════════
def generate_pytorch_code(config: dict) -> str:
    model_name = config.get("model_preset", "Custom")
    if model_name == "Custom":
        model_name = f"{config['model_params']}B model"
    task_type = config["task_type"]
    precision = config["precision"]
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]
    num_gpus = config["num_gpus"]
    dataset_tokens = config["dataset_tokens"]
    epochs = config["epochs"]
    model_params = config["model_params"]
    grad_checkpoint = config.get("grad_checkpoint", False)
    deepspeed_stage = config.get("deepspeed_stage", "None")
    is_lora = task_type == "LoRA / QLoRA"
    uses_ds = deepspeed_stage != "None"
    prec_str = "bf16" if precision == "FP16/BF16" else "fp32" if precision == "FP32" else precision.lower()
    torch_dtype = "bfloat16" if prec_str == "bf16" else "float16" if prec_str == "fp16" else "float32"
    lr = "2e-4" if is_lora else ("3e-4" if task_type == "Pretraining" else "2e-5")

    code = f'''"""
{"=" * 55}
  GPU Resource Profiler -- {model_name}
  Task: {task_type} | Precision: {precision}
  Generated by GPU Resource Estimator
{"=" * 55}
"""
import torch
import time
import json
from dataclasses import dataclass
'''
    if is_lora:
        code += "from peft import LoraConfig, get_peft_model, TaskType\n"
    if uses_ds:
        code += "import deepspeed\n"
    code += "from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments\n"
    if num_gpus > 1 and not uses_ds:
        code += "from torch.distributed.fsdp import FullyShardedDataParallel as FSDP\n"

    hf_model_id = f'meta-llama/{config.get("model_preset", "LLaMA-3-8B").replace(" ", "-")}' if config.get("model_preset") != "Custom" else "your-model-id"

    code += f'''
# --- Configuration ---
@dataclass
class TrainingConfig:
    model_name: str = "{hf_model_id}"
    batch_size: int = {batch_size}
    seq_length: int = {seq_length}
    precision: str = "{prec_str}"
    gradient_checkpointing: bool = {grad_checkpoint}
    dataset_tokens: float = {dataset_tokens}e9  # {dataset_tokens}B tokens
    epochs: int = {epochs}
    num_gpus: int = {num_gpus}
    learning_rate: float = {lr}

config = TrainingConfig()

# --- GPU Memory Profiling ---
def profile_gpu_memory():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return {{
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "total_gpu_gb": torch.cuda.get_device_properties(0).total_mem / 1024**3,
        "device_name": torch.cuda.get_device_name(0),
    }}

# --- FLOPS Measurement ---
def measure_flops_per_step(model, dummy_input, num_warmup=3, num_measure=10):
    device = next(model.parameters()).device
    model.train()
    for _ in range(num_warmup):
        with torch.cuda.amp.autocast(dtype=torch.{torch_dtype}):
            outputs = model(**dummy_input)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].mean()
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    timings = []
    for _ in range(num_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.cuda.amp.autocast(dtype=torch.{torch_dtype}):
            outputs = model(**dummy_input)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].mean()
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return sum(timings) / len(timings), timings

def estimate_total_flops(model_params_B, seq_len, batch_size, total_tokens):
    """Kaplan/Chinchilla: C ~ 6 * N * D"""
    N = model_params_B * 1e9
    flops_per_token = 6 * N
    total = flops_per_token * total_tokens
    return total, flops_per_token
'''

    if is_lora:
        code += '''
# --- LoRA Configuration ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
'''

    if uses_ds:
        stage_num = 3 if "3" in deepspeed_stage else (2 if "2" in deepspeed_stage else 1)
        prec_key = "bf16" if prec_str == "bf16" else "fp16"
        code += f'''
# --- DeepSpeed Config ({deepspeed_stage}) ---
ds_config = {{
    "train_batch_size": {batch_size * num_gpus},
    "train_micro_batch_size_per_gpu": {batch_size},
    "gradient_accumulation_steps": 1,
    "{prec_key}": {{"enabled": True}},
    "zero_optimization": {{
        "stage": {stage_num},
        "overlap_comm": True,
        "reduce_scatter": True,
        "contiguous_gradients": True,
    }},
    "gradient_clipping": 1.0,
    "wall_clock_breakdown": True,
}}
'''

    code += f'''
# --- Main Profiling Pipeline ---
if __name__ == "__main__":
    print("=" * 60)
    print(f"  GPU Resource Profiler -- {model_name}")
    print(f"  Task: {task_type}")
    print("=" * 60)

    assert torch.cuda.is_available(), "CUDA GPU required"
    mem_before = profile_gpu_memory()
    print(f"\\n[GPU] {{mem_before['device_name']}}")
    print(f"[GPU] Total VRAM: {{mem_before['total_gpu_gb']:.1f}} GB")

    total_tokens = config.dataset_tokens * config.epochs
    peak_tflops = 312  # Adjust for your GPU

    total_flops, fpt = estimate_total_flops({model_params}, config.seq_length, config.batch_size, total_tokens)
    effective_tflops = peak_tflops * config.num_gpus * 0.40 * 1e12
    est_time_h = total_flops / effective_tflops / 3600

    print(f"\\n[Theoretical Estimate]")
    print(f"  Total FLOPs: {{total_flops:.2e}}")
    print(f"  Estimated time: {{est_time_h:.1f}} hours")

    print(f"\\n[Loading model...]")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.{torch_dtype},
    ).cuda()
'''
    if is_lora:
        code += "    model = get_peft_model(model, lora_config)\n    model.print_trainable_parameters()\n"
    if grad_checkpoint:
        code += "    model.gradient_checkpointing_enable()\n"

    code += f'''
    mem_loaded = profile_gpu_memory()
    print(f"  Memory after loading: {{mem_loaded['allocated_gb']:.2f}} GB")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dummy = tokenizer(
        ["benchmark " * (config.seq_length // 2)] * config.batch_size,
        max_length=config.seq_length, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to("cuda")
    dummy["labels"] = dummy["input_ids"].clone()

    print(f"\\n[Measuring real step time...]")
    avg_time, timings = measure_flops_per_step(model, dummy)

    mem_peak = profile_gpu_memory()
    tokens_per_step = config.batch_size * config.seq_length
    flops_per_step = 6 * {model_params} * 1e9 * tokens_per_step
    achieved_tflops = flops_per_step / avg_time / 1e12
    mfu = achieved_tflops / peak_tflops

    print(f"  Avg time/step: {{avg_time*1000:.1f}} ms")
    print(f"  Throughput: {{tokens_per_step/avg_time:.0f}} tokens/s")
    print(f"  Achieved TFLOPS: {{achieved_tflops:.1f}}")
    print(f"  MFU: {{mfu*100:.1f}}%")
    print(f"  Peak memory: {{mem_peak['max_allocated_gb']:.2f}} GB")

    results = {{
        "gpu": mem_before["device_name"],
        "model": config.model_name,
        "task": "{task_type}",
        "vram_peak_gb": mem_peak["max_allocated_gb"],
        "achieved_tflops": achieved_tflops,
        "mfu": mfu,
        "avg_step_time_ms": avg_time * 1000,
        "throughput_tokens_per_sec": tokens_per_step / avg_time,
    }}
    with open("gpu_profile_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\\n[Results saved to gpu_profile_results.json]")
    print("=" * 60)
'''
    return code


def generate_tf_code(config: dict) -> str:
    model_name = config.get("model_preset", "Custom")
    if model_name == "Custom":
        model_name = f"{config['model_params']}B model"
    precision = config["precision"]
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]
    num_gpus = config["num_gpus"]
    model_params = config["model_params"]
    prec_str = "mixed_bfloat16" if precision == "FP16/BF16" else ("float32" if precision == "FP32" else "mixed_float16")

    code = f'''"""
{"=" * 55}
  GPU Resource Profiler (TensorFlow/Keras) -- {model_name}
  Task: {config["task_type"]} | Precision: {precision}
  Generated by GPU Resource Estimator
{"=" * 55}
"""
import tensorflow as tf
import time
import json
import numpy as np

tf.keras.mixed_precision.set_global_policy("{prec_str}")

BATCH_SIZE = {batch_size}
SEQ_LENGTH = {seq_length}
MODEL_PARAMS_B = {model_params}
NUM_GPUS = {num_gpus}

def profile_gpu_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU detected")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    info = tf.config.experimental.get_memory_info("GPU:0")
    return {{
        "current_mb": info["current"] / 1024**2,
        "peak_mb": info["peak"] / 1024**2,
        "num_gpus": len(gpus),
    }}

def benchmark_step(model, dummy_x, dummy_y, optimizer, num_warmup=3, num_runs=10):
    @tf.function(jit_compile=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for _ in range(num_warmup):
        train_step(dummy_x, dummy_y)
    timings = []
    for _ in range(num_runs):
        tf.test.experimental.sync_devices()
        start = time.perf_counter()
        train_step(dummy_x, dummy_y)
        tf.test.experimental.sync_devices()
        timings.append(time.perf_counter() - start)
    return np.mean(timings), timings
'''
    if num_gpus > 1:
        code += '''
strategy = tf.distribute.MirroredStrategy()
print(f"GPUs: {strategy.num_replicas_in_sync}")
'''

    code += f'''
if __name__ == "__main__":
    print("=" * 60)
    print(f"  GPU Resource Profiler (TF) -- {model_name}")
    print("=" * 60)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\\nGPUs detected: {{len(gpus)}}")

    inputs = tf.keras.Input(shape=(SEQ_LENGTH,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(32000, 256)(inputs)
    for _ in range(4):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ff = tf.keras.layers.Dense(1024, activation="gelu")(x)
        ff = tf.keras.layers.Dense(256)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)
    output = tf.keras.layers.Dense(32000)(x)
    bench_model = tf.keras.Model(inputs, output)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    dummy_x = tf.random.uniform((BATCH_SIZE, SEQ_LENGTH), 0, 32000, dtype=tf.int32)
    dummy_y = tf.random.uniform((BATCH_SIZE, SEQ_LENGTH), 0, 32000, dtype=tf.int32)

    avg_time, timings = benchmark_step(bench_model, dummy_x, dummy_y, optimizer)
    mem = profile_gpu_memory()
    print(f"  Time/step: {{avg_time*1000:.1f}} ms")
    print(f"  Peak memory: {{mem['peak_mb']:.0f}} MB")
    print("=" * 60)
'''
    return code


# ═══════════════════════════════════════════════════════════════════
# DISPLAY (Rich or plain)
# ═══════════════════════════════════════════════════════════════════
def _make_vram_bar(used: float, total: float, width: int = 30) -> str:
    pct = min(used / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {pct*100:.0f}%"


def display_results_rich(
    estimation: EstimationResult,
    time_est: Optional[TrainingTimeResult],
    config: dict,
    gpu_spec: dict,
    console: "Console",
):
    gpu_name = config["gpu"]
    num_gpus = config["num_gpus"]
    total_vram = gpu_spec["vram"] * num_gpus
    fits = estimation.total_vram <= total_vram
    gpus_needed = math.ceil(estimation.total_vram / gpu_spec["vram"])
    is_training = TASK_TYPES[config["task_type"]]["is_training"]

    # ─── Header ───
    console.print()
    console.print(Panel(
        f"[bold green]GPU Resource Estimator v2[/]\n"
        f"[dim]Model: {config.get('model_preset', 'Custom')} ({config['model_params']}B) | "
        f"Task: {config['task_type']} | GPU: {gpu_name} x{num_gpus}[/]",
        border_style="green",
    ))

    # ─── VRAM Result ───
    status_color = "green" if fits else "red"
    status_icon = "PASS" if fits else "FAIL"
    status_text = f"Feasible with {num_gpus} GPU(s)" if fits else f"Insufficient -- min. {gpus_needed} GPU(s) needed"

    vram_table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    vram_table.add_column(style="bold")
    vram_table.add_column(style="bold " + status_color, justify="right")
    vram_table.add_row("VRAM Required", format_bytes(estimation.total_vram))
    vram_table.add_row("VRAM Available", format_bytes(total_vram))
    vram_table.add_row("Usage", _make_vram_bar(estimation.total_vram, total_vram))
    vram_table.add_row("Status", f"[{status_color}]{status_icon} - {status_text}[/]")

    if is_training and time_est:
        vram_table.add_row("Est. Training Time", f"[bold cyan]{format_time(time_est.training_time_seconds)}[/]")
        cost_hr = gpu_spec["price_hr"] * max(num_gpus, gpus_needed)
        total_cost = cost_hr * (time_est.training_time_seconds / 3600)
        vram_table.add_row("Est. Cost", f"[yellow]{format_cost(total_cost)}[/] ({format_cost(cost_hr)}/h)")

    console.print(Panel(vram_table, title="[bold]VRAM & Feasibility[/]", border_style=status_color))

    # ─── Memory Breakdown ───
    mem_table = Table(box=box.ROUNDED, title="Memory Breakdown")
    mem_table.add_column("Component", style="cyan")
    mem_table.add_column("Size", justify="right", style="bold")
    mem_table.add_column("%", justify="right", style="dim")

    total_mem = estimation.total_vram
    components = [("Model Weights", estimation.weights_gb)]
    if is_training:
        components.append(("Gradients", estimation.gradients_gb))
        components.append(("Optimizer (Adam)", estimation.optimizer_gb))
        components.append(("Activations", estimation.activations_gb))
    else:
        components.append(("KV Cache", estimation.kv_cache_gb))

    for name, val in components:
        pct = (val / total_mem * 100) if total_mem > 0 else 0
        mem_table.add_row(name, format_bytes(val), f"{pct:.0f}%")

    console.print(mem_table)

    # ─── Training Metrics ───
    if is_training and time_est:
        perf_table = Table(box=box.ROUNDED, title="Training Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", justify="right", style="bold")

        perf_table.add_row("Total FLOPs", format_flops(estimation.total_flops))
        perf_table.add_row("FLOPs/token", format_flops(estimation.flops_per_token))
        perf_table.add_row("Throughput", f"{time_est.throughput_tokens_per_sec:,.0f} tokens/s")
        perf_table.add_row("Total Steps", f"{estimation.total_steps:,}")
        perf_table.add_row("Time/Step", f"{time_est.time_per_step * 1000:.1f} ms")
        perf_table.add_row("MFU", f"{time_est.effective_mfu * 100:.1f}%")
        perf_table.add_row("Scaling Efficiency", f"{time_est.scaling_efficiency * 100:.1f}%")
        perf_table.add_row("Effective TFLOPS", f"{time_est.effective_tflops_total:.1f} / {time_est.peak_tflops} peak")

        console.print(perf_table)

    # ─── Formulas ───
    if is_training and time_est:
        formula_lines = [
            f"[yellow]Compute (Kaplan/Chinchilla):[/] C = 6 x N x D = {format_flops(estimation.total_flops)}",
            f"[yellow]Attention overhead:[/] 12 x L x H x S^2 = 12 x {estimation.num_layers} x {estimation.hidden_size} x {config['seq_length']}^2",
            f"[yellow]MFU:[/] {time_est.effective_tflops_total:.1f} / {time_est.peak_tflops} = {time_est.effective_mfu*100:.1f}%",
            f"[yellow]Multi-GPU scaling:[/] {'NVLink' if gpu_spec['nvlink'] else 'PCIe'} -> {time_est.scaling_efficiency*100:.1f}% efficiency",
            f"[yellow]Time = C / (GPUs x Peak x MFU x eta):[/] {format_time(time_est.training_time_seconds)}",
        ]
        console.print(Panel("\n".join(formula_lines), title="[bold yellow]Formulas Used[/]", border_style="yellow"))

    # ─── Recommendations ───
    recs = _generate_recommendations(estimation, time_est, config, gpu_spec, total_vram, fits, gpus_needed)
    if recs:
        rec_text = "\n".join(recs)
        console.print(Panel(rec_text, title="[bold]Recommendations[/]", border_style="blue"))
    console.print()


def display_results_plain(
    estimation: EstimationResult,
    time_est: Optional[TrainingTimeResult],
    config: dict,
    gpu_spec: dict,
):
    gpu_name = config["gpu"]
    num_gpus = config["num_gpus"]
    total_vram = gpu_spec["vram"] * num_gpus
    fits = estimation.total_vram <= total_vram
    gpus_needed = math.ceil(estimation.total_vram / gpu_spec["vram"])
    is_training = TASK_TYPES[config["task_type"]]["is_training"]

    print("\n" + "=" * 60)
    print(f"  GPU Resource Estimator v2")
    print(f"  Model: {config.get('model_preset', 'Custom')} ({config['model_params']}B)")
    print(f"  Task: {config['task_type']} | GPU: {gpu_name} x{num_gpus}")
    print("=" * 60)

    status = "PASS" if fits else "FAIL"
    print(f"\n  VRAM Required:  {format_bytes(estimation.total_vram)}")
    print(f"  VRAM Available: {format_bytes(total_vram)}")
    print(f"  Usage:          {_make_vram_bar(estimation.total_vram, total_vram)}")
    if fits:
        print(f"  Status:         {status} - Feasible with {num_gpus} GPU(s)")
    else:
        print(f"  Status:         {status} - Insufficient, min. {gpus_needed} GPU(s) needed")

    if is_training and time_est:
        cost_hr = gpu_spec["price_hr"] * max(num_gpus, gpus_needed)
        total_cost = cost_hr * (time_est.training_time_seconds / 3600)
        print(f"\n  --- Training Metrics ---")
        print(f"  Training Time:     {format_time(time_est.training_time_seconds)}")
        print(f"  Total FLOPs:       {format_flops(estimation.total_flops)}")
        print(f"  Throughput:        {time_est.throughput_tokens_per_sec:,.0f} tokens/s")
        print(f"  Total Steps:       {estimation.total_steps:,}")
        print(f"  Time/Step:         {time_est.time_per_step * 1000:.1f} ms")
        print(f"  MFU:               {time_est.effective_mfu * 100:.1f}%")
        print(f"  Scaling Eff:       {time_est.scaling_efficiency * 100:.1f}%")
        print(f"  Effective TFLOPS:  {time_est.effective_tflops_total:.1f} / {time_est.peak_tflops} peak")
        print(f"  Est. Cost:         {format_cost(total_cost)} ({format_cost(cost_hr)}/h)")

    # Memory breakdown
    print(f"\n  --- Memory Breakdown ---")
    print(f"  Weights:     {format_bytes(estimation.weights_gb)}")
    if is_training:
        print(f"  Gradients:   {format_bytes(estimation.gradients_gb)}")
        print(f"  Optimizer:   {format_bytes(estimation.optimizer_gb)}")
        print(f"  Activations: {format_bytes(estimation.activations_gb)}")
    else:
        print(f"  KV Cache:    {format_bytes(estimation.kv_cache_gb)}")

    recs = _generate_recommendations(estimation, time_est, config, gpu_spec, total_vram, fits, gpus_needed)
    if recs:
        print(f"\n  --- Recommendations ---")
        for r in recs:
            # Strip rich markup for plain mode
            clean = r.replace("[red]", "").replace("[yellow]", "").replace("[blue]", "")
            clean = clean.replace("[green]", "").replace("[/]", "")
            print(f"  {clean}")
    print()


def _generate_recommendations(estimation, time_est, config, gpu_spec, total_vram, fits, gpus_needed):
    recs = []
    task_type = config["task_type"]
    precision = config["precision"]
    grad_checkpoint = config.get("grad_checkpoint", False)
    deepspeed_stage = config.get("deepspeed_stage", "None")
    num_gpus = config["num_gpus"]
    is_training = TASK_TYPES[task_type]["is_training"]
    utilization_pct = (estimation.total_vram / total_vram) * 100 if total_vram > 0 else 0

    if not fits:
        recs.append(f"[red]X Insufficient memory. Min. {gpus_needed} GPU(s) required, or quantize further.[/]")
        if task_type == "Full Fine-tuning":
            recs.append("[yellow]-> LoRA/QLoRA reduces ~90% memory footprint -- strongly recommended.[/]")
        if precision == "FP32":
            recs.append("[yellow]-> FP16/BF16 halves memory with minimal quality loss.[/]")
        if not grad_checkpoint and is_training:
            recs.append("[yellow]-> Gradient checkpointing saves ~70% activation memory.[/]")
        if deepspeed_stage == "None" and num_gpus > 1:
            recs.append("[yellow]-> DeepSpeed ZeRO Stage 2/3 partitions optimizer+gradients across GPUs.[/]")
    else:
        if utilization_pct < 40:
            recs.append("[blue]i GPU underutilized. Increase batch size for better throughput.[/]")
        elif utilization_pct > 85:
            recs.append("[yellow]! Tight margin. OOM risk with longer sequences.[/]")
        else:
            recs.append("[green]OK Balanced configuration. Good utilization/margin ratio.[/]")

    if time_est:
        if time_est.effective_mfu < 0.30:
            recs.append(f"[yellow]! Low MFU ({time_est.effective_mfu*100:.0f}%). Check data pipeline and I/O.[/]")
        if time_est.training_time_seconds > 86400 * 30:
            recs.append("[blue]i Training > 1 month. Consider more GPUs or a smaller model.[/]")
        if num_gpus > 1 and not gpu_spec["nvlink"]:
            recs.append("[yellow]! PCIe without NVLink: ~12% scaling loss per GPU doubling.[/]")

    return recs


def display_scaling_table(config: dict, gpu_spec: dict, estimation: EstimationResult, console=None):
    gpu_configs = [1, 2, 4, 8, 16, 32, 64]
    is_training = TASK_TYPES[config["task_type"]]["is_training"]
    if not is_training:
        return

    rows = []
    for n in gpu_configs:
        est = estimate_resources(
            model_params=config["model_params"],
            task_type=config["task_type"],
            batch_size=config["batch_size"],
            seq_length=config["seq_length"],
            precision=config["precision"],
            grad_checkpoint=config.get("grad_checkpoint", False),
            deepspeed_stage=config.get("deepspeed_stage", "None"),
            num_gpus=n,
            dataset_tokens=config["dataset_tokens"] * 1e9,
            epochs=config["epochs"],
            num_layers=config.get("num_layers", 32),
            hidden_size=config.get("hidden_size", 4096),
        )
        t = estimate_training_time(est, gpu_spec, n, config["precision"], config.get("deepspeed_stage", "None"), config.get("framework", "PyTorch + DeepSpeed"))
        cost = gpu_spec["price_hr"] * n * (t.training_time_seconds / 3600)
        rows.append({
            "gpus": n,
            "time": format_time(t.training_time_seconds),
            "cost": format_cost(cost),
            "efficiency": f"{t.scaling_efficiency * 100:.0f}%",
            "tflops": f"{t.effective_tflops_total:.0f}",
            "tok_s": f"{t.throughput_tokens_per_sec:,.0f}",
            "raw_cost": cost,
            "raw_hours": t.training_time_seconds / 3600,
        })

    if console and HAS_RICH:
        table = Table(box=box.ROUNDED, title="Multi-GPU Scaling Projection")
        table.add_column("GPUs", style="bold cyan", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Cost", justify="right", style="yellow")
        table.add_column("Efficiency", justify="right")
        table.add_column("TFLOPS", justify="right")
        table.add_column("tok/s", justify="right")

        # Find cheapest config under 1 year
        valid = [r for r in rows if r["raw_hours"] < 24 * 365]
        best_cost = min(valid, key=lambda r: r["raw_cost"])["raw_cost"] if valid else None

        for r in rows:
            style = ""
            if best_cost is not None and r["raw_cost"] == best_cost:
                style = "bold green"
            if r["gpus"] == config["num_gpus"]:
                style = "bold white on grey23"
            table.add_row(f"{r['gpus']}x", r["time"], r["cost"], r["efficiency"], r["tflops"], r["tok_s"], style=style)

        console.print(table)
        console.print("[dim]Green = best cost/time ratio. Highlighted = current config.[/]")
    else:
        print("\n  --- Multi-GPU Scaling ---")
        print(f"  {'GPUs':>5}  {'Time':>12}  {'Cost':>10}  {'Eff':>6}  {'TFLOPS':>7}  {'tok/s':>10}")
        for r in rows:
            marker = " <--" if r["gpus"] == config["num_gpus"] else ""
            print(f"  {r['gpus']:>4}x  {r['time']:>12}  {r['cost']:>10}  {r['efficiency']:>6}  {r['tflops']:>7}  {r['tok_s']:>10}{marker}")


def display_gpu_comparison(config: dict, estimation: EstimationResult, console=None):
    is_training = TASK_TYPES[config["task_type"]]["is_training"]
    if not is_training:
        return

    rows = []
    for gpu_name, spec in GPU_DATABASE.items():
        needed = math.ceil(estimation.total_vram / spec["vram"])
        t = estimate_training_time(estimation, spec, needed, config["precision"], config.get("deepspeed_stage", "None"), config.get("framework", "PyTorch + DeepSpeed"))
        cost = spec["price_hr"] * needed * (t.training_time_seconds / 3600)
        rows.append({
            "name": gpu_name,
            "needed": needed,
            "time": format_time(t.training_time_seconds),
            "cost": format_cost(cost),
            "raw_cost": cost,
        })

    rows.sort(key=lambda r: r["raw_cost"])

    if console and HAS_RICH:
        table = Table(box=box.ROUNDED, title="GPU Comparison (for this config)")
        table.add_column("GPU", style="cyan")
        table.add_column("GPUs Needed", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Cost", justify="right", style="yellow")

        for r in rows:
            style = "bold green" if r["name"] == config["gpu"] else ""
            gpu_style = "green" if r["needed"] <= 8 else "red"
            table.add_row(r["name"], f"[{gpu_style}]{r['needed']}[/]", r["time"], r["cost"], style=style)

        console.print(table)
    else:
        print("\n  --- GPU Comparison ---")
        print(f"  {'GPU':<30}  {'#GPUs':>5}  {'Time':>12}  {'Cost':>10}")
        for r in rows:
            marker = " <--" if r["name"] == config["gpu"] else ""
            print(f"  {r['name']:<30}  {r['needed']:>5}  {r['time']:>12}  {r['cost']:>10}{marker}")


# ═══════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════════════
def _select_from_list(prompt_text: str, options: list, default: str = None) -> str:
    if HAS_QUESTIONARY:
        result = questionary.select(prompt_text, choices=options, default=default).ask()
        if result is None:
            sys.exit(0)
        return result
    else:
        print(f"\n{prompt_text}")
        for i, opt in enumerate(options):
            marker = " (default)" if opt == default else ""
            print(f"  [{i+1}] {opt}{marker}")
        while True:
            raw = input(f"  Choice [1-{len(options)}]: ").strip()
            if not raw and default:
                return default
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            print("  Invalid choice, try again.")


def _input_number(prompt_text: str, default, min_val=None, max_val=None):
    while True:
        raw = input(f"  {prompt_text} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Must be <= {max_val}")
                continue
            return int(val) if val == int(val) else val
        except ValueError:
            print("  Invalid number.")


def _input_bool(prompt_text: str, default: bool) -> bool:
    if HAS_QUESTIONARY:
        return questionary.confirm(prompt_text, default=default).ask()
    default_str = "Y/n" if default else "y/N"
    raw = input(f"  {prompt_text} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def interactive_menu() -> dict:
    console = Console() if HAS_RICH else None

    if console:
        console.print(Panel("[bold green]GPU Resource Estimator v2[/]\n[dim]Interactive Configuration[/]", border_style="green"))
    else:
        print("\n" + "=" * 50)
        print("  GPU Resource Estimator v2 - Interactive Mode")
        print("=" * 50)

    # 1. Model
    model_preset = _select_from_list("Select model preset:", list(MODEL_PRESETS.keys()), default="LLaMA 3 8B")

    if model_preset == "Custom":
        model_params = _input_number("Model parameters (billions)", 7, 0.01, 2000)
        num_layers = int(_input_number("Number of layers", 32, 1, 200))
        hidden_size = int(_input_number("Hidden size", 4096, 64, 32768))
    else:
        preset = MODEL_PRESETS[model_preset]
        model_params = preset["params"]
        num_layers = preset["layers"]
        hidden_size = preset["hidden"]
        print(f"  -> {model_params}B params, {num_layers} layers, hidden={hidden_size}")

    # 2. Task
    task_type = _select_from_list("Select task type:", list(TASK_TYPES.keys()), default="Full Fine-tuning")
    is_training = TASK_TYPES[task_type]["is_training"]
    print(f"  -> {TASK_TYPES[task_type]['description']}")

    # 3. GPU
    gpu = _select_from_list("Select GPU:", list(GPU_DATABASE.keys()), default="NVIDIA A100 (80GB)")
    num_gpus = int(_input_number("Number of GPUs", 1, 1, 1024))

    # 4. Training params
    batch_size = int(_input_number("Batch size", 4, 1, 512))
    seq_length = int(_input_number("Sequence length", 2048, 128, 131072))
    precision = _select_from_list("Precision:", ["FP32", "FP16/BF16", "INT8", "INT4"], default="FP16/BF16")

    grad_checkpoint = False
    deepspeed_stage = "None"
    dataset_tokens = 10
    epochs = 1
    framework = "PyTorch + DeepSpeed"

    if is_training:
        grad_checkpoint = _input_bool("Enable gradient checkpointing?", False)
        deepspeed_stage = _select_from_list("DeepSpeed ZeRO stage:", list(DEEPSPEED_STAGES.keys()), default="None")
        dataset_tokens = _input_number("Dataset size (billion tokens)", 10, 0.001, 15000)
        epochs = int(_input_number("Epochs", 1, 1, 100))
        framework = _select_from_list("Framework:", list(FRAMEWORKS.keys()), default="PyTorch + DeepSpeed")

    return {
        "model_preset": model_preset,
        "model_params": model_params,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "task_type": task_type,
        "gpu": gpu,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "precision": precision,
        "grad_checkpoint": grad_checkpoint,
        "deepspeed_stage": deepspeed_stage,
        "dataset_tokens": dataset_tokens,
        "epochs": epochs,
        "framework": framework,
    }


def run_estimation(config: dict, export_path: Optional[str] = None, show_scaling: bool = True, show_comparison: bool = True, generate_code_type: Optional[str] = None):
    console = Console() if HAS_RICH else None
    gpu_spec = GPU_DATABASE[config["gpu"]]
    is_training = TASK_TYPES[config["task_type"]]["is_training"]

    estimation = estimate_resources(
        model_params=config["model_params"],
        task_type=config["task_type"],
        batch_size=config["batch_size"],
        seq_length=config["seq_length"],
        precision=config["precision"],
        grad_checkpoint=config.get("grad_checkpoint", False),
        deepspeed_stage=config.get("deepspeed_stage", "None"),
        num_gpus=config["num_gpus"],
        dataset_tokens=config["dataset_tokens"] * 1e9,
        epochs=config["epochs"],
        num_layers=config.get("num_layers", 32),
        hidden_size=config.get("hidden_size", 4096),
    )

    time_est = None
    if is_training:
        time_est = estimate_training_time(
            estimation, gpu_spec, config["num_gpus"],
            config["precision"], config.get("deepspeed_stage", "None"),
            config.get("framework", "PyTorch + DeepSpeed"),
        )

    # Display results
    if console:
        display_results_rich(estimation, time_est, config, gpu_spec, console)
    else:
        display_results_plain(estimation, time_est, config, gpu_spec)

    # Scaling table
    if show_scaling and is_training:
        display_scaling_table(config, gpu_spec, estimation, console)

    # GPU comparison
    if show_comparison and is_training:
        display_gpu_comparison(config, estimation, console)

    # Code generation
    if generate_code_type:
        if generate_code_type == "tensorflow":
            code = generate_tf_code(config)
        else:
            code = generate_pytorch_code(config)
        if console:
            from rich.syntax import Syntax
            console.print(Panel(Syntax(code, "python", theme="monokai"), title=f"[bold green]Generated {generate_code_type.title()} Code[/]", border_style="green"))
        else:
            print(f"\n--- Generated {generate_code_type.title()} Code ---")
            print(code)

    # Export
    if export_path:
        total_vram = gpu_spec["vram"] * config["num_gpus"]
        gpus_needed = math.ceil(estimation.total_vram / gpu_spec["vram"])
        export_data = {
            "config": config,
            "estimation": {
                "vram_required_gb": round(estimation.total_vram, 2),
                "vram_available_gb": total_vram,
                "fits": estimation.total_vram <= total_vram,
                "gpus_needed": gpus_needed,
                "model_size_gb": round(estimation.model_size_gb, 2),
                "total_flops": estimation.total_flops,
                "total_steps": estimation.total_steps,
                "memory_breakdown": {
                    "weights_gb": round(estimation.weights_gb, 2),
                    "gradients_gb": round(estimation.gradients_gb, 2),
                    "optimizer_gb": round(estimation.optimizer_gb, 2),
                    "activations_gb": round(estimation.activations_gb, 2),
                    "kv_cache_gb": round(estimation.kv_cache_gb, 2),
                },
            },
        }
        if time_est:
            export_data["training_time"] = {
                "seconds": round(time_est.training_time_seconds, 1),
                "formatted": format_time(time_est.training_time_seconds),
                "throughput_tokens_per_sec": round(time_est.throughput_tokens_per_sec, 0),
                "mfu": round(time_est.effective_mfu, 4),
                "scaling_efficiency": round(time_est.scaling_efficiency, 4),
                "effective_tflops": round(time_est.effective_tflops_total, 1),
                "cost_total": round(gpu_spec["price_hr"] * max(config["num_gpus"], gpus_needed) * (time_est.training_time_seconds / 3600), 2),
            }
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        msg = f"Results exported to {export_path}"
        if console:
            console.print(f"[bold green]{msg}[/]")
        else:
            print(msg)

    return estimation, time_est


# ═══════════════════════════════════════════════════════════════════
# CLI (argparse)
# ═══════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU Resource Estimator v2 - Estimate VRAM, training time, FLOPs for AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                              # Interactive mode
  %(prog)s --model "LLaMA 3 8B" --task "Full Fine-tuning" --gpu "NVIDIA A100 (80GB)"
  %(prog)s --model "LLaMA 3 70B" --task "LoRA / QLoRA" --gpu "NVIDIA RTX 4090 (24GB)" --num-gpus 4
  %(prog)s --params 13 --task "Pretraining" --gpu "NVIDIA H100 (80GB)" --num-gpus 8 --export results.json
  %(prog)s --list-gpus
  %(prog)s --list-models
  %(prog)s --list-tasks
""",
    )
    parser.add_argument("--model", type=str, help="Model preset name (e.g. 'LLaMA 3 8B')")
    parser.add_argument("--params", type=float, help="Custom model size in billions of parameters")
    parser.add_argument("--layers", type=int, help="Number of layers (for custom model)")
    parser.add_argument("--hidden", type=int, help="Hidden size (for custom model)")
    parser.add_argument("--task", type=str, help="Task type (e.g. 'Full Fine-tuning')")
    parser.add_argument("--gpu", type=str, help="GPU name (e.g. 'NVIDIA A100 (80GB)')")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--precision", type=str, default="FP16/BF16", choices=["FP32", "FP16/BF16", "INT8", "INT4"])
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--deepspeed", type=str, default="None", help="DeepSpeed ZeRO stage")
    parser.add_argument("--dataset-tokens", type=float, default=10, help="Dataset size in billion tokens")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--framework", type=str, default="PyTorch + DeepSpeed", help="Training framework")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument("--generate-code", type=str, choices=["pytorch", "tensorflow"], help="Generate training code")
    parser.add_argument("--no-scaling", action="store_true", help="Skip multi-GPU scaling table")
    parser.add_argument("--no-comparison", action="store_true", help="Skip GPU comparison table")
    parser.add_argument("--list-gpus", action="store_true", help="List all supported GPUs")
    parser.add_argument("--list-models", action="store_true", help="List all model presets")
    parser.add_argument("--list-tasks", action="store_true", help="List all task types")
    return parser


def list_gpus():
    console = Console() if HAS_RICH else None
    if console:
        table = Table(box=box.ROUNDED, title="Supported GPUs")
        table.add_column("GPU", style="cyan")
        table.add_column("VRAM", justify="right")
        table.add_column("FP16 TF", justify="right")
        table.add_column("FP32 TF", justify="right")
        table.add_column("BW GB/s", justify="right")
        table.add_column("TDP W", justify="right")
        table.add_column("$/hr", justify="right", style="yellow")
        table.add_column("NVLink", justify="center")
        table.add_column("Arch", style="dim")
        for name, spec in GPU_DATABASE.items():
            table.add_row(
                name, f"{spec['vram']}GB", f"{spec['tflops_fp16']}", f"{spec['tflops_fp32']}",
                f"{spec['bandwidth']}", f"{spec['tdp']}", f"${spec['price_hr']:.2f}",
                "Yes" if spec["nvlink"] else "No", spec["arch"],
            )
        console.print(table)
    else:
        print(f"\n{'GPU':<30} {'VRAM':>6} {'FP16':>6} {'FP32':>6} {'BW':>6} {'$/hr':>6} {'NVLink':>7} {'Arch'}")
        print("-" * 100)
        for name, spec in GPU_DATABASE.items():
            nvl = "Yes" if spec["nvlink"] else "No"
            print(f"{name:<30} {spec['vram']:>4}GB {spec['tflops_fp16']:>6} {spec['tflops_fp32']:>6} {spec['bandwidth']:>6} ${spec['price_hr']:>5.2f} {nvl:>7} {spec['arch']}")


def list_models():
    console = Console() if HAS_RICH else None
    if console:
        table = Table(box=box.ROUNDED, title="Model Presets")
        table.add_column("Model", style="cyan")
        table.add_column("Params (B)", justify="right")
        table.add_column("Layers", justify="right")
        table.add_column("Hidden", justify="right")
        for name, spec in MODEL_PRESETS.items():
            if name == "Custom":
                continue
            table.add_row(name, f"{spec['params']}", f"{spec['layers']}", f"{spec['hidden']}")
        console.print(table)
    else:
        print(f"\n{'Model':<30} {'Params(B)':>10} {'Layers':>7} {'Hidden':>7}")
        print("-" * 60)
        for name, spec in MODEL_PRESETS.items():
            if name == "Custom":
                continue
            print(f"{name:<30} {spec['params']:>10} {spec['layers']:>7} {spec['hidden']:>7}")


def list_tasks():
    console = Console() if HAS_RICH else None
    if console:
        table = Table(box=box.ROUNDED, title="Task Types")
        table.add_column("Task", style="cyan")
        table.add_column("Type", justify="center")
        table.add_column("Mem Mult", justify="right")
        table.add_column("FLOPs Mult", justify="right")
        table.add_column("Description")
        for name, spec in TASK_TYPES.items():
            task_type = "Train" if spec["is_training"] else "Infer"
            style = "green" if spec["is_training"] else "blue"
            table.add_row(name, f"[{style}]{task_type}[/]", f"{spec['multiplier']}x", f"{spec['flops_multiplier']}x", spec["description"])
        console.print(table)
    else:
        print(f"\n{'Task':<35} {'Type':>6} {'MemX':>5} {'FlopsX':>6} Description")
        print("-" * 100)
        for name, spec in TASK_TYPES.items():
            t = "Train" if spec["is_training"] else "Infer"
            print(f"{name:<35} {t:>6} {spec['multiplier']:>5}x {spec['flops_multiplier']:>5}x {spec['description']}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = build_parser()
    args = parser.parse_args()

    # List commands
    if args.list_gpus:
        list_gpus()
        return
    if args.list_models:
        list_models()
        return
    if args.list_tasks:
        list_tasks()
        return

    # Non-interactive mode (CLI arguments provided)
    if args.model or args.params:
        if args.model:
            if args.model not in MODEL_PRESETS:
                print(f"Error: Unknown model '{args.model}'. Use --list-models to see options.")
                sys.exit(1)
            preset = MODEL_PRESETS[args.model]
            model_params = preset["params"]
            num_layers = preset["layers"]
            hidden_size = preset["hidden"]
        else:
            model_params = args.params
            num_layers = args.layers or max(1, round(math.sqrt(model_params)))
            hidden_size = args.hidden or max(512, round(math.sqrt(model_params * 1e9 / num_layers)))

        task_type = args.task or "Full Fine-tuning"
        if task_type not in TASK_TYPES:
            print(f"Error: Unknown task '{task_type}'. Use --list-tasks to see options.")
            sys.exit(1)

        gpu = args.gpu or "NVIDIA A100 (80GB)"
        if gpu not in GPU_DATABASE:
            print(f"Error: Unknown GPU '{gpu}'. Use --list-gpus to see options.")
            sys.exit(1)

        config = {
            "model_preset": args.model or "Custom",
            "model_params": model_params,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "task_type": task_type,
            "gpu": gpu,
            "num_gpus": args.num_gpus,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "precision": args.precision,
            "grad_checkpoint": args.grad_checkpoint,
            "deepspeed_stage": args.deepspeed,
            "dataset_tokens": args.dataset_tokens,
            "epochs": args.epochs,
            "framework": args.framework,
        }

        run_estimation(
            config,
            export_path=args.export,
            show_scaling=not args.no_scaling,
            show_comparison=not args.no_comparison,
            generate_code_type=args.generate_code,
        )
    else:
        # Interactive mode
        config = interactive_menu()
        print()

        # Ask what to show
        show_scaling = True
        show_comparison = True
        generate_code_type = None

        if TASK_TYPES[config["task_type"]]["is_training"]:
            show_scaling = _input_bool("Show multi-GPU scaling table?", True)
            show_comparison = _input_bool("Show GPU comparison?", True)

        gen_code = _input_bool("Generate profiling code?", False)
        if gen_code:
            generate_code_type = _select_from_list("Framework for code:", ["pytorch", "tensorflow"], default="pytorch")

        export_path = None
        do_export = _input_bool("Export results to JSON?", False)
        if do_export:
            export_path = input("  Export path [results.json]: ").strip() or "results.json"

        run_estimation(
            config,
            export_path=export_path,
            show_scaling=show_scaling,
            show_comparison=show_comparison,
            generate_code_type=generate_code_type,
        )


if __name__ == "__main__":
    main()
