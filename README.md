




- End-to-end MoE training stack built on PyTorch, AMP, and Hugging Face datasets.
- Custom Muon optimizer (`optimizers/muon.py`) plus hybrid Muon+AdamW scheduling for stability.
- Pluggable dataset/tokenizer configs and streaming-friendly data pipeline.
- Experiment harness for optimizer sweeps, logging, plotting, and checkpointing.
- Reproducible write-up sources in `paper.tex` / `paper.pdf`.

## Repository Layout
- `train_moe.py` – CLI entry point that wires configs, dataset prep, training, and checkpoint export.
- `configs/` – dataclass-based configs (`moe_config.py`, `dataset_config.py`) shared across scripts.
- `data/` – tokenizer setup, streaming dataset utilities, and DataLoader helpers.
- `models/` – minimal MoE LLM building blocks (`components.py`, `layers.py`, `moe_llm.py`).
- `optimizers/` – Muon optimizer implementation and related math kernels.
- `training/` – trainer loop, evaluation helpers, metric plotting, and logging utilities.
- `experiments/exp1_muon_vs_adam/` – scripts for Muon vs AdamW sweeps and reporting.
- `utils/` – miscellaneous helpers (seed control, logging setup).

