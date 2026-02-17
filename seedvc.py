import importlib.resources
import os
from typing import Union, Tuple

import torch
import yaml
import numpy as np
import soundfile as sf
from pathlib import Path

# Audio input: path string or (audio_array, sample_rate) tuple
AudioInput = Union[str, Tuple[np.ndarray, int], Tuple[torch.Tensor, int]]


class SeedVC:
    """Voice conversion using Seed-VC V2 model."""

    def __init__(
        self,
        ar_checkpoint_path: str = None,
        cfm_checkpoint_path: str = None,
        compile: bool = False,
        device: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the voice conversion model.

        Args:
            ar_checkpoint_path: Path to custom AR checkpoint
            cfm_checkpoint_path: Path to custom CFM checkpoint
            compile: Whether to compile the model for faster inference
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            dtype: Data type for inference
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self._model = self._load_model(ar_checkpoint_path, cfm_checkpoint_path, compile)

    def _get_config_path(self):
        return importlib.resources.files("configs.v2") / "vc_wrapper.yaml"

    def _load_model(self, ar_checkpoint_path, cfm_checkpoint_path, compile):
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        cfg = DictConfig(yaml.safe_load(open(self._get_config_path(), "r")))
        model = instantiate(cfg)
        model.load_checkpoints(
            ar_checkpoint_path=ar_checkpoint_path,
            cfm_checkpoint_path=cfm_checkpoint_path
        )
        model.to(self.device)
        model.eval()
        model.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=self.dtype, device=self.device)

        if compile:
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            if hasattr(torch._inductor.config, "fx_graph_cache"):
                torch._inductor.config.fx_graph_cache = True
            model.compile_ar()

        return model

    def convert(
        self,
        source: AudioInput,
        target: AudioInput,
        output: str = None,
        diffusion_steps: int = 100,
        length_adjust: float = 1.0,
        intelligibility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        convert_style: bool = False,
        anonymization_only: bool = False,
    ):
        """
        Convert voice from source audio to match target speaker.

        Args:
            source: Source audio - path string or (audio_array, sample_rate) tuple
            target: Target/reference audio - path string or (audio_array, sample_rate) tuple
            output: Output path for converted audio (optional, returns audio if None)
            diffusion_steps: Number of diffusion steps
            length_adjust: Length adjustment factor
            intelligibility_cfg_rate: CFG rate for intelligibility
            similarity_cfg_rate: CFG rate for similarity
            top_p: Top-p sampling parameter
            temperature: Temperature for sampling
            repetition_penalty: Repetition penalty
            convert_style: Convert style/emotion/accent
            anonymization_only: Anonymization only mode

        Returns:
            If output is None: tuple of (sample_rate, audio_array)
            If output is provided: output path
        """
        generator = self._model.convert_voice_with_streaming(
            source_audio=source,
            target_audio=target,
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            intelligebility_cfg_rate=intelligibility_cfg_rate,
            similarity_cfg_rate=similarity_cfg_rate,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            convert_style=convert_style,
            anonymization_only=anonymization_only,
            device=self.device,
            dtype=self.dtype,
            stream_output=True
        )

        for _, full_audio in generator:
            pass

        if full_audio is None:
            raise RuntimeError("Failed to convert voice")

        sr, audio = full_audio

        if output:
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            sf.write(output, audio, sr)
            return output

        return sr, audio
