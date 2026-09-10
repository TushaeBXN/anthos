# Stage 1 Run Notes — sft tier, ~195M tokens (~20% Chinchilla)

**Contingency:** If loss curve is rough OR prerouter_kl doesn't trend below 0.3 by step ~2–3k, rerun with PreRouterHead disabled as an ablation before debugging further. Do not debug the combined system in place.

**What "success" means:** Loss drops and plateaus, ACT halt probs stable between 0.1–0.9, MoE routing stays spread (moe_top1 < 50%), prerouter_kl < 0.3 by mid-run. Shallow/hallucinating outputs are expected at this token budget — not bugs.

**Architecture active this run:** PreRouterHead YES, halt_probs YES, LoRA recovery NO (distill.py only).

**Padding fraction (SFT dataloader):** _fill in after running `python3 check_padding_fraction.py --n-batches 100`_
— record the actual number here, e.g. "94% non-pad on 100 batches, Stage 1 SFT set, verdict: fine"
