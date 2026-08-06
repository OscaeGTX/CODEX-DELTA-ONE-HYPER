"""
ai_engine package initializer and high-level runtime orchestration.

This module exposes core classes and provides a hyper-scaled AIEngine lifecycle manager
to initialize, run, and shut down AI components safely and efficiently in production.

Design goals:
- Safe defaults, typed configuration, and explicit lifecycle for deterministic behavior.
- Async-friendly with a sync adapter for legacy code.
- Plugin discovery for modular extensions (preprocessing, postprocessing, detectors, rules).
- Resource-aware initialization (multi-GPU, TPU, accelerators) with graceful fallbacks.
- Lightweight telemetry hooks (Prometheus, StatsD) if available.
- Minimal external hard dependencies: optional imports handled gracefully.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import pkgutil
import signal
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# Re-exported components from submodules (existing in package)
from .ai_model import AIModel  # type: ignore
from .data_preprocessing import DataPreprocessor  # type: ignore
from .model_training import ModelTrainer  # type: ignore
from .inference import InferenceEngine  # type: ignore
from .utils import load_config, load_secure_config  # type: ignore

__all__ = [
    "AIModel",
    "DataPreprocessor",
    "ModelTrainer",
    "InferenceEngine",
    "AIEngine",
    "setup_ai_engine",
    "discover_plugins",
    "EngineConfig",
    "load_config",
    "load_secure_config",
]

_LOG = logging.getLogger("ai_engine")
_DEFAULT_LOG_LEVEL = logging.INFO

# ---------- Configuration dataclass ----------


@dataclass
class EngineConfig:
    """Typed configuration for the AI engine runtime."""

    config_path: Optional[str] = None
    device: Optional[str] = None  # 'cpu' | 'cuda' | 'auto' | 'mps' | 'tpu'
    num_workers: int = 4
    distributed_backend: Optional[str] = None  # 'torch.distributed', 'deepspeed', 'accelerate'
    enable_telemetry: bool = False
    telemetry_backend: Optional[str] = None  # 'prometheus' | 'statsd' | None
    telemetry_port: int = 8000
    plugins: Sequence[str] = field(default_factory=list)
    secure_config_verification: bool = True
    log_level: int = _DEFAULT_LOG_LEVEL
    seed: Optional[int] = None
    allow_insecure_fallback: bool = False  # If secure verification fails, fallback if True


# ---------- Utilities ----------


def _init_logging(level: int) -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
    _LOG.debug("Logging initialized at level %s", level)


def _detect_device(preferred: Optional[str] = None) -> str:
    """Detect best available device. Soft imports to avoid hard deps."""
    if preferred and preferred != "auto":
        return preferred

    # Try CUDA (PyTorch), then JAX/TPU, then MPS, else CPU
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    # JAX check (TPU)
    try:
        import jax  # type: ignore

        # jax.devices() will reveal TPU/GPU/CPU
        devices = jax.devices()
        if any("tpu" in str(d).lower() for d in devices):
            return "tpu"
        if any("gpu" in str(d).lower() for d in devices):
            return "gpu"
    except Exception:
        pass

    return "cpu"


def _safe_import(module_name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        _LOG.debug("Optional module '%s' could not be imported: %s", module_name, exc)
        return None


def discover_plugins(package: str = "ai_engine.plugins") -> List[str]:
    """Discover importable plugin modules in the ai_engine.plugins namespace.

    Returns list of fully-qualified module names.
    """
    found: List[str] = []
    try:
        pkg = importlib.import_module(package)
    except Exception:
        _LOG.debug("Plugin package %s not found", package)
        return found

    if not hasattr(pkg, "__path__"):
        return found

    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        _LOG.debug("Discovered plugin: %s (ispkg=%s)", name, ispkg)
        found.append(name)
    return found


# ---------- Engine Lifecycle Manager ----------


class AIEngine:
    """High-level engine lifecycle manager for the ai_engine package.

    Use as a context manager (sync or async) to ensure proper startup and teardown.
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        _init_logging(self.config.log_level)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = False
        self._bg_tasks: List[asyncio.Task] = []
        self._plugins: Dict[str, ModuleType] = {}
        self.ai_model: Optional[AIModel] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.trainer: Optional[ModelTrainer] = None
        self.inference: Optional[InferenceEngine] = None
        self._shutdown_event = asyncio.Event()

    async def _load_config(self) -> Dict[str, Any]:
        """Load and verify config using package utils; fall back to safe defaults."""
        cfg_path = self.config.config_path
        try:
            if self.config.secure_config_verification:
                # try secure loader if available
                loaded = await asyncio.to_thread(load_secure_config, cfg_path)  # type: ignore
            else:
                loaded = await asyncio.to_thread(load_config, cfg_path)  # type: ignore
            _LOG.info("Configuration loaded from %s", cfg_path or "default")
            return loaded or {}
        except Exception as exc:
            _LOG.warning("Failed to load secure config: %s", exc)
            if self.config.allow_insecure_fallback:
                try:
                    loaded = await asyncio.to_thread(load_config, cfg_path)  # type: ignore
                    _LOG.warning("Fell back to insecure config loader")
                    return loaded or {}
                except Exception as exc2:
                    _LOG.error("Fallback config load failed: %s", exc2)
            raise

    async def _initialize_components(self, cfg: Dict[str, Any]) -> None:
        """Initialize AIModel, Preprocessor, Trainer, Inference components safely."""
        # Lazy import to reduce startup cost
        self.preprocessor = DataPreprocessor(cfg.get("preprocessing", {}))
        self.ai_model = AIModel(cfg.get("model", {}))
        self.trainer = ModelTrainer(self.ai_model, cfg.get("training", {}))
        self.inference = InferenceEngine(self.ai_model, cfg.get("inference", {}))
        _LOG.info("Core components initialized: Preprocessor, Model, Trainer, Inference")

    async def _apply_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        try:
            import random
            import numpy as np  # type: ignore

            random.seed(seed)
            np.random.seed(seed)
            # Torch seed if available
            try:
                import torch  # type: ignore

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            _LOG.info("Global RNG seed set to %s", seed)
        except Exception as exc:
            _LOG.debug("Could not set full seed: %s", exc)

    async def _start_telemetry(self) -> None:
        if not self.config.enable_telemetry:
            return
        backend = (self.config.telemetry_backend or "prometheus").lower()
        if backend == "prometheus":
            prometheus_client = _safe_import("prometheus_client")
            if prometheus_client:
                try:
                    from prometheus_client import start_http_server  # type: ignore

                    start_http_server(self.config.telemetry_port)
                    _LOG.info("Prometheus metrics server started on port %s", self.config.telemetry_port)
                except Exception as exc:
                    _LOG.warning("Failed to start Prometheus server: %s", exc)
            else:
                _LOG.warning("prometheus_client not installed; telemetry disabled.")
        else:
            _LOG.debug("Telemetry backend '%s' not implemented in this runtime.", backend)

    async def start(self) -> "AIEngine":
        if self._started:
            return self
        # Event loop setup if not present
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # Device detection
        self.config.device = _detect_device(self.config.device)
        _LOG.info("Engine selected device: %s", self.config.device)

        # Load config (possibly secure)
        cfg = await self._load_config()

        # Apply deterministic seed if present
        await self._apply_seed(self.config.seed or cfg.get("seed"))

        # Initialize core components with config
        await self._initialize_components(cfg)

        # Discover and load plugins if any
        plugin_modules = list(self.config.plugins) or discover_plugins()
        for mod_name in plugin_modules:
            try:
                mod = importlib.import_module(mod_name)
                # If plugin exposes `register(engine)` call it
                if hasattr(mod, "register") and inspect.isfunction(getattr(mod, "register")):
                    await asyncio.to_thread(mod.register, self)  # allow sync registration
                self._plugins[mod_name] = mod
                _LOG.info("Loaded plugin: %s", mod_name)
            except Exception as exc:
                _LOG.warning("Failed to load plugin %s: %s", mod_name, exc)

        # Start telemetry if configured
        await self._start_telemetry()

        self._started = True
        _LOG.info("AIEngine started successfully")
        return self

    async def stop(self) -> None:
        if not self._started:
            return
        # Trigger background task cancellation
        for task in list(self._bg_tasks):
            task.cancel()
        # Allow components to perform graceful shutdown
        try:
            if self.trainer and hasattr(self.trainer, "shutdown"):
                await asyncio.to_thread(self.trainer.shutdown)  # type: ignore
            if self.inference and hasattr(self.inference, "shutdown"):
                await asyncio.to_thread(self.inference.shutdown)  # type: ignore
        except Exception as exc:
            _LOG.warning("Error while shutting down components: %s", exc)

        self._started = False
        self._shutdown_event.set()
        _LOG.info("AIEngine stopped")

    async def wait_shutdown(self) -> None:
        await self._shutdown_event.wait()

    def create_background_task(self, coro: asyncio.coroutine) -> asyncio.Task:
        if self._loop is None:
            raise RuntimeError("Engine event loop is not initialized")
        task = asyncio.create_task(coro)
        self._bg_tasks.append(task)
        return task

    # Context manager support (async and sync)
    async def __aenter__(self) -> "AIEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def __enter__(self) -> "AIEngine":
        # Provide sync context manager by running the event loop to start
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.start())
        finally:
            # keep loop for background tasks if any; do not close here
            loop.close()

    def __exit__(self, exc_type, exc, tb) -> None:
        # Attempt to stop the engine synchronously
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.stop())
        finally:
            loop.close()


# ---------- Convenience setup function ----------


async
