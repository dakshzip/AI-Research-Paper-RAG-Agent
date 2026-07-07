"""Admin script: download a curated set of landmark AI papers from arXiv.

Fills the corpus folder with ~80 influential papers (classics through 2025)
alongside the ones already present, then index them with:

    python -m scripts.download_papers [--dir "AI Research Papers"]
    python -m scripts.seed_corpus --dir "AI Research Papers"

Downloads are skipped when the target PDF already exists, so re-running is
safe and only fetches what's missing.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "AI Research Papers"

# Filename (without .pdf) -> arXiv ID. Names follow the existing corpus style:
# "Short Name - Descriptive Title".
PAPERS: dict[str, str] = {
    # --- LLMs and foundation models ---
    "Transformer-XL - Attentive Language Models Beyond Fixed-Length Context": "1901.02860",
    "XLNet - Generalized Autoregressive Pretraining": "1906.08237",
    "ELECTRA - Pre-training Text Encoders as Discriminators": "2003.10555",
    "DistilBERT - Distilled Version of BERT": "1910.01108",
    "BART - Denoising Sequence-to-Sequence Pre-training": "1910.13461",
    "Sentence-BERT - Sentence Embeddings using Siamese Networks": "1908.10084",
    "GPT-4 Technical Report": "2303.08774",
    "PaLM - Scaling Language Modeling with Pathways": "2204.02311",
    "Chinchilla - Training Compute-Optimal Large Language Models": "2203.15556",
    "Llama 2 - Open Foundation and Fine-Tuned Chat Models": "2307.09288",
    "Llama 3 - The Llama 3 Herd of Models": "2407.21783",
    "Mistral 7B": "2310.06825",
    "Mixtral of Experts": "2401.04088",
    "Gemini - A Family of Highly Capable Multimodal Models": "2312.11805",
    "Qwen2.5 Technical Report": "2412.15115",
    "DeepSeek-V3 Technical Report": "2412.19437",
    "DeepSeek-R1 - Incentivizing Reasoning via RL": "2501.12948",
    "Phi-3 Technical Report": "2404.14219",
    "Gemma - Open Models from Gemini Research": "2403.08295",
    "Mamba - Linear-Time Sequence Modeling with Selective State Spaces": "2312.00752",
    # --- Alignment and instruction tuning ---
    "FLAN - Finetuned Language Models Are Zero-Shot Learners": "2109.01652",
    "Flan-PaLM - Scaling Instruction-Finetuned Language Models": "2210.11416",
    "Self-Instruct - Aligning LMs with Self-Generated Instructions": "2212.10560",
    "LoRA - Low-Rank Adaptation of Large Language Models": "2106.09685",
    "QLoRA - Efficient Finetuning of Quantized LLMs": "2305.14314",
    "Deep RL from Human Preferences": "1706.03741",
    "DPO - Direct Preference Optimization": "2305.18290",
    "Constitutional AI - Harmlessness from AI Feedback": "2212.08073",
    "LIMA - Less Is More for Alignment": "2305.11206",
    # --- Reasoning and agents ---
    "Zero-Shot CoT - LLMs are Zero-Shot Reasoners": "2205.11916",
    "Self-Consistency Improves Chain of Thought": "2203.11171",
    "Tree of Thoughts - Deliberate Problem Solving with LLMs": "2305.10601",
    "ReAct - Synergizing Reasoning and Acting in LMs": "2210.03629",
    "Toolformer - LMs Can Teach Themselves to Use Tools": "2302.04761",
    "Reflexion - Language Agents with Verbal RL": "2303.11366",
    "Generative Agents - Interactive Simulacra of Human Behavior": "2304.03442",
    "STaR - Self-Taught Reasoner": "2203.14465",
    "Let's Verify Step by Step": "2305.20050",
    "Scaling LLM Test-Time Compute Optimally": "2408.03314",
    # --- Retrieval and RAG ---
    "DPR - Dense Passage Retrieval for Open-Domain QA": "2004.04906",
    "ColBERT - Efficient Passage Search via Late Interaction": "2004.12832",
    "REALM - Retrieval-Augmented Language Model Pre-Training": "2002.08909",
    "RETRO - Improving LMs by Retrieving from Trillions of Tokens": "2112.04426",
    "HyDE - Precise Zero-Shot Dense Retrieval": "2212.10496",
    "Self-RAG - Learning to Retrieve, Generate, and Critique": "2310.11511",
    "RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval": "2401.18059",
    "GraphRAG - From Local to Global Query-Focused Summarization": "2404.16130",
    "RAG Survey - Retrieval-Augmented Generation for LLMs": "2312.10997",
    "Lost in the Middle - How LMs Use Long Contexts": "2307.03172",
    # --- Efficiency and architecture ---
    "FlashAttention - Fast and Memory-Efficient Exact Attention": "2205.14135",
    "FlashAttention-2 - Faster Attention with Better Parallelism": "2307.08691",
    "RoFormer - Rotary Position Embedding": "2104.09864",
    "GQA - Grouped-Query Attention": "2305.13245",
    "Switch Transformers - Scaling to Trillion Parameter Models": "2101.03961",
    "MoE - Outrageously Large Neural Networks": "1701.06538",
    "vLLM - Efficient LLM Serving with PagedAttention": "2309.06180",
    "GPTQ - Accurate Post-Training Quantization": "2210.17323",
    "LLM.int8 - 8-bit Matrix Multiplication for Transformers": "2208.07339",
    "Knowledge Distillation - Distilling Knowledge in a Neural Network": "1503.02531",
    # --- Vision and generative models ---
    "EfficientNet - Rethinking Model Scaling for CNNs": "1905.11946",
    "Mask R-CNN": "1703.06870",
    "DETR - End-to-End Object Detection with Transformers": "2005.12872",
    "Swin Transformer - Hierarchical ViT using Shifted Windows": "2103.14030",
    "MAE - Masked Autoencoders Are Scalable Vision Learners": "2111.06377",
    "SimCLR - Contrastive Learning of Visual Representations": "2002.05709",
    "SAM - Segment Anything": "2304.02643",
    "Latent Diffusion - High-Resolution Image Synthesis (Stable Diffusion)": "2112.10752",
    "NeRF - Representing Scenes as Neural Radiance Fields": "2003.08934",
    "VAE - Auto-Encoding Variational Bayes": "1312.6114",
    # --- Multimodal ---
    "Flamingo - A Visual Language Model for Few-Shot Learning": "2204.14198",
    "BLIP-2 - Bootstrapping Language-Image Pre-training": "2301.12597",
    "LLaVA - Visual Instruction Tuning": "2304.08485",
    "Whisper - Robust Speech Recognition via Weak Supervision": "2212.04356",
    # --- RL and evaluation ---
    "PPO - Proximal Policy Optimization": "1707.06347",
    "MuZero - Mastering Games Without Knowing the Rules": "1911.08265",
    "Decision Transformer - RL via Sequence Modeling": "2106.01345",
    "Emergent Abilities of Large Language Models": "2206.07682",
    "MMLU - Measuring Massive Multitask Language Understanding": "2009.03300",
    "GSM8K - Training Verifiers to Solve Math Word Problems": "2110.14168",
    "Codex - Evaluating LLMs Trained on Code": "2107.03374",
    "SWE-bench - Can LMs Resolve Real-World GitHub Issues": "2310.06770",
}

USER_AGENT = "SCA-RAG-corpus-builder (mailto:dakshtaneja.bifs@gmail.com)"
DELAY_SECONDS = 3.0  # be polite to arXiv
MIN_PDF_BYTES = 10_000  # smaller than this is an error page, not a paper


def download_pdf(arxiv_id: str, dest: Path) -> None:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF") or len(data) < MIN_PDF_BYTES:
        raise ValueError(f"response is not a valid PDF ({len(data)} bytes)")
    dest.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download curated arXiv papers.")
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help="Destination folder for PDFs (default: %(default)s).",
    )
    args = parser.parse_args()
    dest_dir = Path(args.dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = 0, 0, []
    for index, (name, arxiv_id) in enumerate(sorted(PAPERS.items()), start=1):
        dest = dest_dir / f"{name}.pdf"
        prefix = f"[{index}/{len(PAPERS)}]"
        if dest.exists():
            print(f"{prefix} skip (exists): {dest.name}")
            skipped += 1
            continue
        try:
            download_pdf(arxiv_id, dest)
            size_mb = dest.stat().st_size / 1e6
            print(f"{prefix} ok ({size_mb:.1f} MB): {dest.name}")
            downloaded += 1
        except Exception as exc:
            print(f"{prefix} FAILED ({arxiv_id}): {dest.name} -> {exc}")
            failed.append((name, arxiv_id))
        time.sleep(DELAY_SECONDS)

    print(f"\nDone. Downloaded {downloaded}, skipped {skipped}, failed {len(failed)}.")
    if failed:
        print("Failed papers (retry by re-running, or fetch manually):")
        for name, arxiv_id in failed:
            print(f"  - {name} (https://arxiv.org/abs/{arxiv_id})")
    return 1 if failed and not downloaded else 0


if __name__ == "__main__":
    raise SystemExit(main())
