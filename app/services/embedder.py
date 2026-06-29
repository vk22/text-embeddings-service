import os
import gc
import threading
from typing import List

import numpy as np


CLAP_CHECKPOINT_URLS = {
    0: "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-best.pt",
    1: "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt",
    2: "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-fusion-best.pt",
    3: "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt",
}

CLAP_CHECKPOINT_NAMES = {
    0: "630k-best.pt",
    1: "630k-audioset-best.pt",
    2: "630k-fusion-best.pt",
    3: "630k-audioset-fusion-best.pt",
}


class EmbedderService:
    def __init__(self, state=None) -> None:
        self.state = state
        self._lock = threading.Lock()

        cache_dir = os.environ.get(
            "CLAP_CACHE_DIR",
            "/app/.cache/revibed-clap",
        )
        os.makedirs(cache_dir, exist_ok=True)

        self._set_stage("cache_dir_ready")
        print(f"step 1: cache_dir={cache_dir}", flush=True)

        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        self._configure_torch_runtime()

        self._set_stage("env_configured")
        print("step 2: env configured", flush=True)

        self.model = self._load_model()

        self._set_stage("checkpoint_loaded")
        print("step 5: checkpoint loaded", flush=True)

    def _set_stage(self, value: str) -> None:
        if self.state is not None:
            self.state["stage"] = value

    def _load_model(self):
        if os.environ.get("CLAP_TEXT_ONLY", "1") == "1":
            return self._load_text_only_model()

        self._set_stage("import_laion_clap")
        print("step 3: importing laion_clap", flush=True)
        import laion_clap

        self._set_stage("create_clap_module")
        print("step 4: creating module", flush=True)
        model = laion_clap.CLAP_Module(enable_fusion=False)

        self._set_stage("load_checkpoint")
        print("step 4.1: loading checkpoint", flush=True)
        ckpt = os.environ.get("CLAP_CHECKPOINT")
        model_id = int(os.environ.get("CLAP_MODEL_ID", "1"))
        model.load_ckpt(ckpt=ckpt, model_id=model_id, verbose=False)

        model.eval()
        self._prune_audio_modules(model)
        self._maybe_quantize_model(model)

        self._set_stage("checkpoint_load_finished")
        print("step 4.2: checkpoint load finished", flush=True)
        return model

    def _load_text_only_model(self):
        self._set_stage("create_text_only_model")
        print("step 3: creating text-only CLAP module", flush=True)

        checkpoint_path = self._resolve_checkpoint_path()
        model = TextOnlyCLAP()

        self._set_stage("load_text_checkpoint")
        print(f"step 4.1: loading text checkpoint from {checkpoint_path}", flush=True)
        model.load_clap_checkpoint(checkpoint_path)

        model.eval()
        self._maybe_quantize_model(model)

        self._set_stage("checkpoint_load_finished")
        print("step 4.2: text checkpoint load finished", flush=True)
        return model

    def _resolve_checkpoint_path(self) -> str:
        ckpt = os.environ.get("CLAP_CHECKPOINT")
        if ckpt:
            return ckpt

        cache_dir = os.environ.get("CLAP_CACHE_DIR", "/app/.cache/revibed-clap")
        model_id = int(os.environ.get("CLAP_MODEL_ID", "1"))
        checkpoint_name = CLAP_CHECKPOINT_NAMES[model_id]
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            return checkpoint_path

        self._set_stage("download_checkpoint")
        print(f"downloading CLAP checkpoint to {checkpoint_path}", flush=True)
        import wget

        os.makedirs(cache_dir, exist_ok=True)
        wget.download(CLAP_CHECKPOINT_URLS[model_id], checkpoint_path)
        print("checkpoint download completed", flush=True)
        return checkpoint_path

    def _configure_torch_runtime(self) -> None:
        threads = int(os.environ.get("CLAP_TORCH_THREADS", "1"))

        try:
            import torch

            torch.set_num_threads(threads)
            torch.set_num_interop_threads(threads)
        except Exception as e:
            print(f"torch runtime configuration skipped: {e}", flush=True)

    def _prune_audio_modules(self, model) -> None:
        if os.environ.get("CLAP_PRUNE_AUDIO", "1") != "1":
            return

        clap_model = getattr(model, "model", None)
        if clap_model is None:
            return

        self._set_stage("prune_audio_modules")
        for attr in ("audio_branch", "audio_projection", "audio_transform"):
            if hasattr(clap_model, attr):
                setattr(clap_model, attr, None)

        gc.collect()
        print("step 4.3: audio modules pruned for text-only service", flush=True)

    def _maybe_quantize_model(self, model) -> None:
        if os.environ.get("CLAP_DYNAMIC_QUANTIZE", "0") != "1":
            return

        self._set_stage("quantize_model")
        try:
            import torch

            model.model = torch.quantization.quantize_dynamic(
                model.model if hasattr(model, "model") else model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            gc.collect()
            print("step 4.4: dynamic int8 quantization applied", flush=True)
        except Exception as e:
            print(f"dynamic quantization skipped: {e}", flush=True)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Empty text query")

        max_chars = int(os.environ.get("CLAP_MAX_TEXT_CHARS", "1000"))
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars]

        with self._lock:
            import torch

            with torch.inference_mode():
                emb = self.model.get_text_embedding([text])

        vector = emb[0]
        vector = np.asarray(vector, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()


class TextOnlyCLAP:
    def __init__(self) -> None:
        import torch
        from torch import nn
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

        config = RobertaConfig(
            max_position_embeddings=514,
            type_vocab_size=1,
        )
        self.model = nn.Module()
        self.model.text_branch = RobertaModel(config)
        self.model.text_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        self._torch = torch

    def eval(self) -> None:
        self.model.eval()

    def load_clap_checkpoint(self, checkpoint_path: str) -> None:
        try:
            checkpoint = self._torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
        except TypeError:
            checkpoint = self._torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        state_dict = checkpoint.get("state_dict", checkpoint)
        text_branch_state = self._extract_prefixed_state(
            state_dict,
            "module.text_branch.",
        )
        text_branch_state.pop("embeddings.position_ids", None)
        text_projection_state = self._extract_prefixed_state(
            state_dict,
            "module.text_projection.",
        )

        self.model.text_branch.load_state_dict(text_branch_state, strict=True)
        self.model.text_projection.load_state_dict(text_projection_state, strict=True)

        del checkpoint
        del state_dict
        gc.collect()

    def get_text_embedding(self, x, use_tensor=False):
        text_input = self.tokenize(
            x,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        output = self.model.text_branch(
            input_ids=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
        )["pooler_output"]
        text_embed = self.model.text_projection(output)
        text_embed = self._torch.nn.functional.normalize(text_embed, dim=-1)
        if use_tensor:
            return text_embed
        return text_embed.detach().cpu().numpy()

    @staticmethod
    def _extract_prefixed_state(state_dict, prefix: str):
        return {
            key.removeprefix(prefix): value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
