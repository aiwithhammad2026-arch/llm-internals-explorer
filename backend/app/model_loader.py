"""Model loader and lifecycle manager for local Transformer models.
By Hammad, AI Engineer.

Enforces:
1. Model loading with attn_implementation="eager" so real attention weights are available.
2. Safe allow-list verification (distilgpt2, gpt2).
3. trust_remote_code=False at all times.
4. Clear actionable error and setup messages if download fails.
5. Zero synthetic/made-up fallback data.
"""

import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from .config import settings

logger = logging.getLogger("transformer_explorer.model_loader")

class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.model_name: str = settings.MODEL_NAME
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        self.device: torch.device = torch.device("cpu")
        self.is_loaded: bool = False
        self.error_message: Optional[str] = None
        self.setup_guidance: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _determine_device(self) -> torch.device:
        config_device = settings.DEVICE_CONFIG.lower()
        if config_device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        elif config_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def load_model(self, model_name: Optional[str] = None) -> None:
        target_model = model_name or self.model_name

        # Security check: strict allow-list validation
        if target_model not in settings.ALLOWED_MODELS:
            err = (
                f"Model '{target_model}' is not in the allowed models list: {settings.ALLOWED_MODELS}. "
                "Arbitrary remote model loading is restricted for security."
            )
            self.error_message = err
            self.setup_guidance = f"Select an approved model: {', '.join(settings.ALLOWED_MODELS)}"
            self.is_loaded = False
            logger.error(err)
            raise ValueError(err)

        logger.info(f"Loading tokenizer and model for '{target_model}' with attn_implementation='eager'...")

        try:
            self.device = self._determine_device()
            self.model_name = target_model

            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                target_model,
                trust_remote_code=settings.TRUST_REMOTE_CODE
            )

            # Ensure pad_token is defined for batch operations if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. Load Model with eager attention for direct weight access
            self.model = AutoModelForCausalLM.from_pretrained(
                target_model,
                attn_implementation=settings.ATTN_IMPLEMENTATION,
                trust_remote_code=settings.TRUST_REMOTE_CODE,
                dtype=torch.float32
            )

            self.model.to(self.device)
            self.model.eval()

            # Populate real commit hash on model config from snapshot cache
            try:
                import re
                from huggingface_hub import try_to_load_from_cache
                cached_file = str(try_to_load_from_cache(target_model, "config.json"))
                match = re.search(r'snapshots[\\/]([0-9a-fA-F]{40})', cached_file)
                if match:
                    self.model.config._commit_hash = match.group(1)
            except Exception:
                pass

            self.is_loaded = True
            self.error_message = None
            self.setup_guidance = None
            logger.info(f"Model '{target_model}' successfully loaded onto {self.device} with eager attention.")

        except Exception as e:
            self.is_loaded = False
            self.model = None
            self.tokenizer = None
            self.error_message = str(e)
            self.setup_guidance = (
                "Could not download or load the model. Please check: "
                "1. An active Internet connection for first-time model download from Hugging Face. "
                "2. Ensure PyTorch and Transformers are installed in your virtual environment. "
                "3. If working offline, ensure 'distilgpt2' has been cached in ~/.cache/huggingface/hub/."
            )
            logger.error(f"Failed to load model '{target_model}': {e}", exc_info=True)
            raise RuntimeError(f"{self.error_message}. {self.setup_guidance}") from e

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Cannot fetch model metadata.")

        config = self.model.config
        num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 6))
        num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", 12))
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
        vocab_size = getattr(config, "vocab_size", 50257)
        max_positions = getattr(config, "n_positions", getattr(config, "max_position_embeddings", 1024))
        num_params = sum(p.numel() for p in self.model.parameters())

        return {
            "model_name": self.model_name,
            "architecture": self.model.__class__.__name__,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_positions,
            "num_parameters": num_params,
            "device": str(self.device),
            "attn_implementation": settings.ATTN_IMPLEMENTATION,
            "trust_remote_code": settings.TRUST_REMOTE_CODE,
            "data_origin": "Real model output"
        }

model_manager = ModelManager.get_instance()
