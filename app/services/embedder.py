import os
from typing import List

import numpy as np


class EmbedderService:
    def __init__(self, state=None) -> None:
        self.state = state

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

        self._set_stage("env_configured")
        print("step 2: env configured", flush=True)

        self.model = self._load_model()

        self._set_stage("checkpoint_loaded")
        print("step 5: checkpoint loaded", flush=True)

    def _set_stage(self, value: str) -> None:
        if self.state is not None:
            self.state["stage"] = value

    def _load_model(self):
        self._set_stage("import_laion_clap")
        print("step 3: importing laion_clap", flush=True)
        import laion_clap

        self._set_stage("create_clap_module")
        print("step 4: creating module", flush=True)
        model = laion_clap.CLAP_Module(enable_fusion=False)

        self._set_stage("load_checkpoint")
        print("step 4.1: loading checkpoint", flush=True)
        model.load_ckpt()

        self._set_stage("checkpoint_load_finished")
        print("step 4.2: checkpoint load finished", flush=True)
        return model

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Empty text query")

        emb = self.model.get_text_embedding([text])

        vector = emb[0]
        vector = np.asarray(vector, dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()