"""
balm_model.py
=============
Shared, importable BALM-PPI model definition + checkpoint loader.

Matched to the balm_ppi_phi checkpoints: LoRA (rank read from file) on target
modules ["query","value"] applied to a SUBSET of layers, plus a cosine
projection head (dim read from file). Everything is read from the checkpoint,
nothing is guessed. Heavy deps import lazily inside build_model().
"""
from __future__ import annotations
import re

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"   # or an absolute local path on an offline node
LORA_ALPHA_TO_RANK_RATIO = 2
LORA_DROPOUT = 0.0
DEFAULT_PROJECTED_SIZE = 128
HEAD_DROPOUT = 0.1


def _torch():
    import torch
    return torch


def _unwrap(sd):
    """Return the tensor state_dict, unwrapping {'state_dict':...} style wrappers."""
    if isinstance(sd, dict) and not any(hasattr(v, "shape") for v in sd.values()):
        for k in ("state_dict", "model", "model_state_dict", "weights"):
            if k in sd:
                return sd[k]
    return sd


def inspect_checkpoint(state_dict):
    """Read the exact LoRA + projection-head architecture from checkpoint keys."""
    modules, layers, ranks = set(), set(), set()
    for k, v in state_dict.items():
        if ".lora_A." in k and k.endswith(".weight"):
            modules.add(k.split(".lora_A.")[0].split(".")[-1])
            ranks.add(v.shape[0])
            m = re.search(r"\.layer\.(\d+)\.", k)
            if m:
                layers.add(int(m.group(1)))
    if not modules:
        raise ValueError("No LoRA ('.lora_A.') weights found in checkpoint.")
    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks not supported here: {sorted(ranks)}")
    r = ranks.pop()
    proj_key = "projection_head.protein_projection.weight"
    if proj_key not in state_dict:
        raise ValueError("projection_head weights missing from checkpoint.")
    return {
        "target_modules": sorted(modules),
        "r": r,
        "lora_alpha": r * LORA_ALPHA_TO_RANK_RATIO,
        "layers_to_transform": sorted(layers),
        "projected_size": state_dict[proj_key].shape[0],
    }


def build_model_classes():
    """Return (BALMProjectionHead, BALMModel). Imports torch/transformers/peft lazily."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    class BALMProjectionHead(nn.Module):
        def __init__(self, embedding_size, projected_size, dropout):
            super().__init__()
            self.protein_projection = nn.Linear(embedding_size, projected_size)
            self.proteina_projection = nn.Linear(embedding_size, projected_size)
            self.dropout = nn.Dropout(dropout)
            self.loss_fn = nn.MSELoss()

        def forward(self, protein_emb, proteina_emb, labels=None):
            p = F.normalize(self.protein_projection(self.dropout(protein_emb)), p=2, dim=1)
            pa = F.normalize(self.proteina_projection(self.dropout(proteina_emb)), p=2, dim=1)
            cos = torch.clamp(F.cosine_similarity(p, pa), -0.9999, 0.9999)
            out = {"cosine_similarity": cos}
            if labels is not None:
                out["loss"] = self.loss_fn(cos, labels)
            return out

    class BALMModel(nn.Module):
        def __init__(self, device, arch):
            super().__init__()
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            base = AutoModel.from_pretrained(MODEL_NAME)
            self.embedding_size = base.config.hidden_size
            self.cls_token = self.tokenizer.cls_token

            cfg = LoraConfig(
                r=arch["r"], lora_alpha=arch["lora_alpha"], lora_dropout=LORA_DROPOUT,
                bias="none", task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=arch["target_modules"],
                layers_to_transform=arch["layers_to_transform"],  # restrict to checkpoint's layers
            )
            # Wrap so parameter names become esm_model.base_model.model.* to match the checkpoint.
            self.esm_model = get_peft_model(base, cfg)
            self.projection_head = BALMProjectionHead(
                self.embedding_size, arch["projected_size"], HEAD_DROPOUT
            )

        def _embed(self, sequences):
            seqs = [s.replace("|", f"{self.cls_token}{self.cls_token}") for s in sequences]
            inputs = self.tokenizer(seqs, return_tensors="pt", padding=True,
                                    truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            hidden = self.esm_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        def forward(self, protein_seqs, proteina_seqs, labels=None):
            return self.projection_head(self._embed(protein_seqs),
                                        self._embed(proteina_seqs), labels)

    return BALMProjectionHead, BALMModel
def _remap_lora_keys(state_dict, prefix="esm_model.base_model.model."):
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("base_model.") and ".lora_" in k:
            suffix = k[len("base_model."):]
            remapped[prefix + suffix] = v
        else:
            remapped[k] = v
    return remapped


def build_model(checkpoint_path, device="cuda"):
    """Build a BALMModel and load `checkpoint_path`, matching its exact architecture."""
    import os
    torch = _torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[balm] WARNING: torch.cuda unavailable; loading BALM on CPU instead of {device}.")
        device = "cpu"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"BALM checkpoint not found: {checkpoint_path}")

    state_dict = _unwrap(torch.load(checkpoint_path, map_location=device))
    arch = inspect_checkpoint(state_dict)
    print(f"[balm] checkpoint arch: r={arch['r']} modules={arch['target_modules']} "
          f"layers={arch['layers_to_transform']} proj={arch['projected_size']}")

    _, BALMModel = build_model_classes()
    model = BALMModel(device=device, arch=arch).to(device)
  
    
    # inside build_model(), after inspect_checkpoint(...):
    state_dict = _remap_lora_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only LoRA + projection-head params must be present; PEFT base weights load from ESM.
    need = [k for k in missing if "lora_" in k or "projection_head" in k]
    got_ckpt = [k for k in unexpected if "lora_" in k or "projection_head" in k]
    if need:
        raise RuntimeError(f"{len(need)} expected trained weights missing after load "
                           f"(first: {need[:3]}). Architecture mismatch?")
    if got_ckpt:
        raise RuntimeError(f"{len(got_ckpt)} trained weights in checkpoint went unused "
                           f"(first: {got_ckpt[:3]}). Key-name mismatch?")
    model.eval()
    print(f"[balm] loaded OK ({len(state_dict)} tensors; "
          f"{len(missing)} base params from ESM, {len(unexpected)} unused).")
    return model


def cos_to_affinity(cos, lo, hi):
    return ((cos + 1) / 2) * (hi - lo) + lo
