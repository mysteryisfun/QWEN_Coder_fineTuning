# Masterplan: Fine-Tuning Qwen3-Coder for Python Unit Test Generation

## Model
- **Base Model**: Qwen/Qwen3-Coder-7B-Instruct (7B parameters, code-specialized; use GGUF Q4_K_S/Q4_K_M for 4GB VRAM RTX 3050).
- **Why Chosen**: Strong community feedback for practical coding; feasible with quantization and CPU offload on low VRAM.
- **Loading**: Use Transformers + bitsandbytes (4-bit quantization, device_map="auto", gradient checkpointing).

## Dataset
- **Name**: CodeRM-UnitTest (Hugging Face: KAKA22/CodeRM-UnitTest).
- **Size**: ~60,000 high-quality synthetic Python unit tests (curated for reward modeling; full size several GBs, streamable).
- **Usage Plan**: Start with 20k–40k subset for pilot; scale to full 60k if stable. Splits: 80/10/10 (train/val/test) via train_test_split.
- **Format**: Pairs of code_under_test + unit_test; tokenize with model's tokenizer, max length 2048 tokens.

## Fine-Tuning Guidelines
- **Method**: QLoRA/PEFT (LoRA rank 8–16; target attention/MLP layers; LR 1e-5–2e-5; warmup 3–5%).
- **Hardware Constraints**: RTX 3050 (4GB VRAM, 16GB RAM) – micro-batch 1–2, accumulation to effective 16–32; epochs 1–2 pilot, up to 2–3 full; early stop on val loss.
- **Objective**: Causal LM on test generation (input: code + prompt; target: unit_test).
- **Evaluation**: Post-training – pytest pass rate, coverage.py, mutation score on held-out samples.
- **Tools**: Hugging Face Datasets/Transformers; stream data to avoid I/O bottlenecks.

## Final Requirements
- **Goal**: A locally runnable model that generates executable Python unit tests from code snippets, fine-tuned on CodeRM-UnitTest for high-quality, assertion-rich outputs.
- **Success Metrics**: >80% pytest pass rate on val set; mutation score >50%; inference latency <10s on laptop.
- **Timeline**: Pilot (1 week); full train (1–2 weeks); iterate based on eval.
- **Risks/Fallback**: If VRAM overflow, subset further or use cloud GPU; fallback to smaller base (e.g., Qwen 3B).
