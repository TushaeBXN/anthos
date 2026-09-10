"""
anthos/bundle.py — Atomic bundle versioning for Anthos checkpoints

A "bundle" is a version-tagged group of three files that must stay in sync:
  1. Base checkpoint     (model weights, optimizer state, step)
  2. LoRA adapter(s)     (delta weights from Chimera pipeline)
  3. PreRouter head      (prerouter_head state dict — saved alongside base ckpt)

Bundles are identified by a version tag (e.g. "mansa_sovereign_v3") and
validated at startup via SHA-256 hashes and weight-shape checks.  If any
component fails validation, the harness rolls back to the last known-good
bundle automatically.

Files written per bundle:
  checkpoints/<run_id>/step_005000.pt                  — base checkpoint
  checkpoints/<run_id>/lora/step_005000_lora.pt        — LoRA delta (optional)
  checkpoints/<run_id>/step_005000_prerouter.pt        — prerouter head weights
  checkpoints/<run_id>/step_005000.bundle.json         — manifest (hashes, shapes, version)
  checkpoints/<run_id>/last_known_good.json            — pointer → last validated bundle

Usage:
    from anthos.bundle import BundleManager

    mgr = BundleManager(run_dir="checkpoints/mansa_sovereign")

    # After saving a checkpoint:
    mgr.save_bundle(model, step, base_ckpt_path, lora_ckpt_path)

    # At inference startup:
    paths = mgr.load_validated_bundle(model, step=None)   # None = latest
    # On hash/shape failure, automatically uses last_known_good.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Manifest dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BundleManifest:
    """Serialisable record of one bundle's identity and integrity data."""
    version_tag:      str
    step:             int
    base_path:        str
    lora_path:        Optional[str]
    prerouter_path:   Optional[str]

    base_sha256:      str
    lora_sha256:      Optional[str]
    prerouter_sha256: Optional[str]

    # Weight shape fingerprints — {param_name: list(shape)}
    base_shapes:      dict = field(default_factory=dict)
    lora_shapes:      dict = field(default_factory=dict)
    prerouter_shapes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BundleManifest":
        d = json.loads(text)
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_fingerprint(state_dict: dict) -> dict:
    return {k: list(v.shape) for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


def _extract_state(loaded) -> dict:
    """Unwrap trainer state dicts that store model under a key."""
    for key in ("model", "model_state_dict"):
        if isinstance(loaded, dict) and key in loaded:
            return loaded[key]
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# BundleManager
# ─────────────────────────────────────────────────────────────────────────────

class BundleManager:
    """
    Saves and validates multi-file Anthos bundles.

    Attach one instance to your training loop (or inference loader).

    Training-side:
        mgr = BundleManager("checkpoints/mansa_sovereign")
        mgr.save_bundle(model, step=5000,
                        base_ckpt_path="checkpoints/mansa_sovereign/step_005000.pt",
                        lora_ckpt_path="checkpoints/mansa_sovereign/lora/step_005000_lora.pt")

    Inference-side:
        paths = mgr.load_validated_bundle(model, step=None)   # None = latest validated
        if paths is None:
            raise RuntimeError("No valid bundle found")
    """

    LAST_GOOD_FILE = "last_known_good.json"

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_bundle(
        self,
        model: nn.Module,
        step:             int,
        base_ckpt_path:   str,
        lora_ckpt_path:   Optional[str]  = None,
        version_tag:      Optional[str]  = None,
    ) -> BundleManifest:
        """
        Save the prerouter head weights, compute all hashes, write the bundle
        manifest, and update last_known_good.json if validation passes.

        The base checkpoint must already exist at base_ckpt_path (call
        torch.save before this).  LoRA checkpoint is optional.

        Returns the manifest.
        """
        # ── PreRouter head ────────────────────────────────────────────────
        prerouter_path: Optional[str] = None
        prerouter_sha:  Optional[str] = None
        prerouter_shapes: dict = {}

        if hasattr(model, "recurrent") and hasattr(model.recurrent, "prerouter_head"):
            prerouter_path = str(Path(base_ckpt_path).with_suffix("")) + "_prerouter.pt"
            prerouter_sd   = model.recurrent.prerouter_head.state_dict()
            torch.save(prerouter_sd, prerouter_path)
            prerouter_sha    = _sha256_file(prerouter_path)
            prerouter_shapes = _shape_fingerprint(prerouter_sd)

        # ── Hashes ────────────────────────────────────────────────────────
        base_sha   = _sha256_file(base_ckpt_path)
        lora_sha   = _sha256_file(lora_ckpt_path) if lora_ckpt_path else None

        # ── Shape fingerprints ────────────────────────────────────────────
        base_state    = _extract_state(torch.load(base_ckpt_path, map_location="cpu", weights_only=True))
        base_shapes   = _shape_fingerprint(base_state)
        lora_shapes   = {}
        if lora_ckpt_path:
            lora_state  = torch.load(lora_ckpt_path, map_location="cpu", weights_only=True)
            lora_shapes = _shape_fingerprint(lora_state)

        manifest = BundleManifest(
            version_tag      = version_tag or f"step_{step:06d}",
            step             = step,
            base_path        = str(base_ckpt_path),
            lora_path        = str(lora_ckpt_path) if lora_ckpt_path else None,
            prerouter_path   = prerouter_path,
            base_sha256      = base_sha,
            lora_sha256      = lora_sha,
            prerouter_sha256 = prerouter_sha,
            base_shapes      = base_shapes,
            lora_shapes      = lora_shapes,
            prerouter_shapes = prerouter_shapes,
        )

        manifest_path = Path(base_ckpt_path).with_suffix(".bundle.json")
        manifest_path.write_text(manifest.to_json())

        # If the manifest validates cleanly, promote it to last-known-good
        if self._validate_manifest(manifest):
            (self.run_dir / self.LAST_GOOD_FILE).write_text(
                json.dumps({"manifest_path": str(manifest_path)}, indent=2)
            )
            print(f"  ✓ Bundle {manifest.version_tag} validated → last_known_good updated")
        else:
            print(f"  ⚠ Bundle {manifest.version_tag} failed validation — last_known_good unchanged")

        return manifest

    # ── Load + validate ───────────────────────────────────────────────────────

    def load_validated_bundle(
        self,
        model:        nn.Module,
        step:         Optional[int] = None,
        strict:       bool          = True,
    ) -> Optional[dict]:
        """
        Load the bundle for `step` (or the latest if step=None) into `model`.
        If hash or shape validation fails, fall back to last_known_good.json.

        Returns a dict with loaded paths on success, or None if no valid bundle
        exists at all.
        """
        manifest = self._resolve_manifest(step)
        if manifest is None:
            print("  ⚠ No bundle manifest found — loading skipped")
            return None

        ok, reason = self._validate_manifest(manifest), "ok"
        if not ok:
            print(f"  ⚠ Bundle {manifest.version_tag} failed validation: rolling back to last-known-good")
            manifest = self._load_last_good_manifest()
            if manifest is None:
                print("  ✗ No last-known-good bundle — cannot load")
                return None
            ok2, _ = self._validate_manifest(manifest), ""
            if not ok2:
                print("  ✗ last-known-good manifest also failed — aborting load")
                return None

        self._apply_bundle_to_model(model, manifest, strict=strict)
        return {
            "version_tag":    manifest.version_tag,
            "step":           manifest.step,
            "base_path":      manifest.base_path,
            "lora_path":      manifest.lora_path,
            "prerouter_path": manifest.prerouter_path,
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_manifest(self, manifest: BundleManifest) -> bool:
        """Returns True only if all present files exist, hashes match, and shapes match."""
        try:
            # Base checkpoint
            if not Path(manifest.base_path).exists():
                return False
            if _sha256_file(manifest.base_path) != manifest.base_sha256:
                return False
            base_state = _extract_state(
                torch.load(manifest.base_path, map_location="cpu", weights_only=True)
            )
            if _shape_fingerprint(base_state) != manifest.base_shapes:
                return False

            # LoRA (optional)
            if manifest.lora_path:
                if not Path(manifest.lora_path).exists():
                    return False
                if _sha256_file(manifest.lora_path) != manifest.lora_sha256:
                    return False
                lora_state = torch.load(manifest.lora_path, map_location="cpu", weights_only=True)
                if _shape_fingerprint(lora_state) != manifest.lora_shapes:
                    return False

            # PreRouter head (optional)
            if manifest.prerouter_path:
                if not Path(manifest.prerouter_path).exists():
                    return False
                if _sha256_file(manifest.prerouter_path) != manifest.prerouter_sha256:
                    return False
                pr_state = torch.load(manifest.prerouter_path, map_location="cpu", weights_only=True)
                if _shape_fingerprint(pr_state) != manifest.prerouter_shapes:
                    return False

            return True

        except Exception:
            return False

    # ── Apply to model ────────────────────────────────────────────────────────

    def _apply_bundle_to_model(
        self,
        model:    nn.Module,
        manifest: BundleManifest,
        strict:   bool = True,
    ) -> None:
        """Load validated weights from manifest into model (in-place)."""
        base_state = _extract_state(
            torch.load(manifest.base_path, map_location="cpu", weights_only=True)
        )
        model.load_state_dict(base_state, strict=strict)

        if manifest.lora_path and Path(manifest.lora_path).exists():
            lora_state = torch.load(manifest.lora_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(lora_state, strict=False)
            # Only LoRA delta keys — expected to be partial

        if (manifest.prerouter_path
                and Path(manifest.prerouter_path).exists()
                and hasattr(model, "recurrent")
                and hasattr(model.recurrent, "prerouter_head")):
            pr_state = torch.load(manifest.prerouter_path, map_location="cpu", weights_only=True)
            model.recurrent.prerouter_head.load_state_dict(pr_state, strict=True)

    # ── Manifest resolution helpers ───────────────────────────────────────────

    def _resolve_manifest(self, step: Optional[int]) -> Optional[BundleManifest]:
        if step is not None:
            p = self.run_dir / f"step_{step:06d}.bundle.json"
            if p.exists():
                return BundleManifest.from_json(p.read_text())
            return None
        # Latest: scan for all bundle manifests, sort by step
        manifests = sorted(self.run_dir.glob("*.bundle.json"))
        if not manifests:
            return None
        return BundleManifest.from_json(manifests[-1].read_text())

    def _load_last_good_manifest(self) -> Optional[BundleManifest]:
        lkg_path = self.run_dir / self.LAST_GOOD_FILE
        if not lkg_path.exists():
            return None
        meta = json.loads(lkg_path.read_text())
        mp   = Path(meta.get("manifest_path", ""))
        if not mp.exists():
            return None
        return BundleManifest.from_json(mp.read_text())
