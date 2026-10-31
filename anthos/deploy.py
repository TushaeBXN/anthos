"""One-command deployment to any platform"""
import subprocess
import json
import os
from pathlib import Path
from typing import Optional


class AnthosDeployer:
    """Deploy to HuggingFace, RunPod, Ollama, or custom server"""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def deploy(self, checkpoint_path: str, platform: str, config: Optional[dict] = None) -> str:
        config = config or {}
        print(f"Preparing {checkpoint_path} for {platform}...")

        if platform == "huggingface":
            return self._deploy_hf(checkpoint_path, config)
        elif platform == "ollama":
            return self._deploy_ollama(checkpoint_path, config)
        elif platform == "runpod":
            return self._deploy_runpod(checkpoint_path, config)
        else:
            raise ValueError(f"Unknown platform '{platform}'. Choose: huggingface | ollama | runpod")

    def _deploy_hf(self, checkpoint_path: str, config: dict) -> str:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        repo_name = f"TushaeBXN/anthos-{config.get('size', '1b')}"
        api = HfApi()
        api.create_repo(repo_name, exist_ok=True)

        output_dir = Path("exports/huggingface")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint to HF export dir
        import shutil
        shutil.copy(checkpoint_path, output_dir / "pytorch_model.bin")

        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_name,
            repo_type="model",
        )

        url = f"https://huggingface.co/{repo_name}"
        print(f"Deployed to {url}")
        return url

    def _deploy_ollama(self, checkpoint_path: str, config: dict) -> str:
        temperature = config.get("temperature", 0.7)
        top_p = config.get("top_p", 0.9)

        modelfile_content = f"""FROM {checkpoint_path}

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "</s>"

SYSTEM "You are Anthos, created by Tushae Thomas in 2026. You are helpful, harmless, and honest."
"""
        with open("Modelfile", "w") as f:
            f.write(modelfile_content)

        subprocess.run(["ollama", "create", "anthos", "-f", "Modelfile"], check=True)
        endpoint = "http://localhost:11434/api/generate"
        print(f"Ollama model 'anthos' created. Endpoint: {endpoint}")
        return endpoint

    def _deploy_runpod(self, checkpoint_path: str, config: dict) -> str:
        dockerfile = f"""FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY anthos /app/anthos
COPY {checkpoint_path} /app/checkpoint.pt
RUN pip install fastapi uvicorn
COPY serve.py /app/serve.py
CMD ["python", "serve.py"]
"""
        with open("Dockerfile", "w") as f:
            f.write(dockerfile)

        subprocess.run(["docker", "build", "-t", "anthos-runpod", "."], check=True)
        print("Docker image 'anthos-runpod' built. Push to RunPod registry to deploy.")
        return "anthos-runpod:latest"


def export_for_huggingface(checkpoint_path: str, output_dir: str = "exports/huggingface"):
    """Standalone export helper"""
    import torch
    from anthos.identity_hardening import CheckpointSigner

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    signer = CheckpointSigner()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    signer.verify(checkpoint)

    torch.save(checkpoint["model_state_dict"], out / "pytorch_model.bin")
    print(f"Exported to {output_dir}/pytorch_model.bin")
