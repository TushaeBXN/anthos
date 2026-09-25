"""
Anthos + Vision — minimal usage example.

Requires: pip install transformers pillow

This downloads a real CLIP checkpoint from the Hugging Face Hub the first
time it runs (small model: ~600MB for clip-vit-base-patch32), then feeds an
image + a text prompt into an Anthos model that has a vision front-end
attached.

At this stage the projector is randomly initialized — this script only
proves the plumbing works end-to-end (shapes, gradients, frozen backbone).
It will NOT produce meaningful captions/answers about the image until the
projector (and ideally the rest of Anthos) has been trained on paired
image-text data — see the two-stage recipe in the project notes:
  Stage A: freeze CLIP + freeze Anthos, train only the projector on
           image-caption pairs (e.g. a subset of LAION or COCO captions).
  Stage B: instruction-tune on visual QA data, optionally unfreezing
           Anthos itself via its existing LoRA machinery.
"""

import torch
from PIL import Image
from transformers import CLIPImageProcessor

from anthos.main   import Anthos, anthos_1b
from anthos.vision import VisionConfig


def main():
    # 1. Build Anthos with a vision front-end attached.
    cfg   = anthos_1b()
    vcfg  = VisionConfig(encoder_name="openai/clip-vit-base-patch32", freeze_encoder=True)
    model = Anthos(cfg, vision_cfg=vcfg)
    model.eval()

    # 2. Preprocess a real image the same way CLIP expects.
    processor = CLIPImageProcessor.from_pretrained(vcfg.encoder_name)
    image     = Image.open("your_image.jpg").convert("RGB")   # replace with a real path
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

    # 3. Tokenize a text prompt with whatever tokenizer Anthos was set up
    #    with (see setup_tokenizer.py in the repo root). Placeholder ids
    #    below stand in for "Describe this image:" — swap in real tokenizer
    #    output once you've loaded one.
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    # 4. Forward pass: image patches are prepended ahead of the text tokens.
    with torch.no_grad():
        logits = model(prompt_ids, n_loops=8, pixel_values=pixel_values)

    print(f"logits shape: {logits.shape}")  # (1, n_image_patches + 8, vocab_size)


if __name__ == "__main__":
    main()
