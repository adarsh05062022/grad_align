import torch
from tqdm import tqdm
from collections import defaultdict


def _accumulate_fisher(
    model,
    forget_dl,
    remain_dl,
    parameters,
    descriptions,
    class_to_forget,
    beta,
    device,
    max_batches=80,
):
    """
    Single pass over the data to accumulate squared gradients (Fisher approximation)
    for both the forget and retain objectives.

    Returns:
        accum_fisher_f : list[Tensor]  – per-parameter forget Fisher  (raw, unnormalized)
        accum_fisher_r : list[Tensor]  – per-parameter retain Fisher  (raw, unnormalized)
    """
    criteria = torch.nn.MSELoss()
    model.eval()

    accum_fisher_f = [torch.zeros_like(p) for p in parameters]
    accum_fisher_r = [torch.zeros_like(p) for p in parameters]

    remain_iter = iter(remain_dl)

    for batch_idx, forget_batch in enumerate(tqdm(forget_dl, desc="[Mask] Accumulating Fisher")):
        if batch_idx >= max_batches:
            break

        forget_images = forget_batch["jpg"].to(device)
        forget_prompts = forget_batch["txt"]

        try:
            remain_batch = next(remain_iter)
        except StopIteration:
            remain_iter = iter(remain_dl)
            remain_batch = next(remain_iter)

        remain_images = remain_batch["jpg"].to(device)
        remain_prompts = remain_batch["txt"]

        pseudo_prompts = ["a photo of a person wearing clothes"] * forget_images.shape[0]

        # ── FORGET Fisher ─────────────────────────────────────────────────────
        forget_batch = {"jpg": forget_images, "txt": forget_prompts}
        pseudo_batch = {"jpg": forget_images, "txt": pseudo_prompts}

        forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
        pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)

        t     = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
        noise = torch.randn_like(forget_input)

        forget_out = model.apply_model(model.q_sample(forget_input, t, noise), t, forget_emb)
        pseudo_out = model.apply_model(model.q_sample(pseudo_input, t, noise), t, pseudo_emb).detach()

        loss_f = criteria(forget_out, pseudo_out) * beta

        grads_f = torch.autograd.grad(loss_f, parameters, retain_graph=False, allow_unused=True)

        for i, g in enumerate(grads_f):
            if g is not None:
                accum_fisher_f[i] += g.detach() ** 2

        del forget_out, pseudo_out, forget_input, pseudo_input, loss_f, grads_f

        # ── RETAIN Fisher ─────────────────────────────────────────────────────
        remain_batch = {"jpg": remain_images, "txt": remain_prompts}
        loss_r = model.shared_step(remain_batch)[0]

        grads_r = torch.autograd.grad(loss_r, parameters, retain_graph=False, allow_unused=True)

        for i, g in enumerate(grads_r):
            if g is not None:
                accum_fisher_r[i] += g.detach() ** 2

        del loss_r, grads_r

    n = min(len(forget_dl), max_batches)
    for i in range(len(accum_fisher_f)):
        accum_fisher_f[i] /= n
        accum_fisher_r[i] /= n

    return accum_fisher_f, accum_fisher_r


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: build importance scores and select top-k mask
# ─────────────────────────────────────────────────────────────────────────────

def _build_mask_from_fisher(
    parameters,
    accum_fisher_f,
    accum_fisher_r,
    target_density,
    lambda_tradeoff,
    importance_variant,
    logger=None,
    param_names=None,   
):
    """
    Given accumulated Fisher tensors, produce a boolean mask.

    Scoring strategy
    ────────────────
    We want parameters where the forget signal is STRONG and the retain
    signal is WEAK.  Two complementary scores are computed then combined:

      ratio  = F_f / (F_r + ε)          ← captures multiplicative dominance
      diff   = F_f_norm - λ·F_r_norm    ← captures additive dominance
                                            after GLOBAL (not per-layer) norm

    Both are computed on the GLOBALLY concatenated vectors so that a tiny
    layer bias cannot outrank a large conv weight just because it happens
    to dominate within its own layer.

    Final score = 0.5 * global_z(ratio) + 0.5 * global_z(diff)
    This is a soft ensemble that is robust to either signal being degenerate.
    """

    # ── Step 1: flatten all Fisher tensors into two global vectors ────────────
    # We deliberately do NOT normalise per-layer before concatenation.
    # Per-layer normalisation would give every layer equal representation,
    # destroying the true global importance signal.

    flat_f_parts, flat_r_parts = [], []
    valid_indices = []        # track which parameter indices contribute

    for i, (f, r) in enumerate(zip(accum_fisher_f, accum_fisher_r)):
        if f is None or r is None:
            continue
        flat_f_parts.append(f.reshape(-1))
        flat_r_parts.append(r.reshape(-1))
        valid_indices.append(i)

    global_f = torch.cat(flat_f_parts)   # shape: [total_params]
    global_r = torch.cat(flat_r_parts)

    # ── Step 2: compute ratio score (globally) ────────────────────────────────
    # ratio captures "how much more does this param matter for forgetting
    # relative to retaining".  A param with F_f=1e-3 and F_r=1e-6 scores
    # much higher than one with F_f=1e3 and F_r=1e3.
    ratio = global_f / (global_r + 1e-10)

    # ── Step 3: compute difference score (globally normalised) ───────────────
    # Normalise GLOBALLY so inter-layer magnitude is preserved.
    f_std = global_f.std().clamp(min=1e-10)
    r_std = global_r.std().clamp(min=1e-10)

    global_f_norm = global_f / f_std
    global_r_norm = global_r / r_std

    diff = global_f_norm - lambda_tradeoff * global_r_norm

    # ── Step 4: Z-score both signals globally, then ensemble ─────────────────
    def global_z(x):
        return (x - x.mean()) / (x.std().clamp(min=1e-10))

    ratio_z = global_z(ratio)
    diff_z = global_z(diff)

    if importance_variant == "ratio":
        score = ratio_z

    elif importance_variant == "difference":
        score = diff_z

    elif importance_variant == "both":
        score = 0.5 * ratio_z + 0.5 * diff_z

    # ── Step 5: deterministic top-k (NO Gumbel noise) ────────────────────────
    # The mask is computed once.  Adding Gumbel noise gives a single random
    # draw that governs the entire training run — pure uncontrolled variance
    # with no benefit.  Deterministic top-k is strictly better here.
    k = max(1, int(target_density * score.numel()))
    top_indices = torch.topk(score, k).indices

    mask_flat = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask_flat[top_indices] = True

    if logger is not None:
        active = mask_flat.sum().item()
        total  = mask_flat.numel()
        logger.info(f"[Mask] density={active/total:.4f}  active={active:,}  total={total:,}")

        # ── Group-level summary instead of per-layer spam ─────────────────────

        group_stats = defaultdict(lambda: {"active": 0, "total": 0, "ff": 0.0, "fr": 0.0, "count": 0})

        offset = 0
        for i, (f, p) in enumerate(zip(accum_fisher_f, parameters)):
            if f is None:
                continue
            n = f.numel()
            layer_active = mask_flat[offset:offset + n].sum().item()
            name = param_names[i] if param_names else f"param[{i:03d}]"

            # ── Determine block location (encoder / middle / decoder) ─────────
            if "input_blocks" in name:
                location = "encoder"
            elif "middle_block" in name:
                location = "middle"
            elif "output_blocks" in name:
                location = "decoder"
            else:
                location = "other"

            # ── Determine layer type ──────────────────────────────────────────
            if "attn2" in name:
                layer_type = "cross-attn"
            elif "attn1" in name:
                layer_type = "self-attn"
            elif "ff." in name or "net." in name:
                layer_type = "feedforward"
            elif "norm" in name:
                layer_type = "norm"
            elif "proj_in" in name or "proj_out" in name:
                layer_type = "proj"
            elif "time_embed" in name:
                layer_type = "time-embed"
            else:
                layer_type = "other"

            group_key = f"{location}|{layer_type}"
            g = group_stats[group_key]
            g["active"] += layer_active
            g["total"]  += n
            g["ff"]     += accum_fisher_f[i].mean().item()
            g["fr"]     += accum_fisher_r[i].mean().item()
            g["count"]  += 1

            offset += n
        # ── Print grouped summary, sorted by density descending ───────────────
        logger.info("[Mask] ── Group Summary (sorted by density) ──────────────────")
        logger.info(f"  {'Group':<28} {'Density':>8}  {'Active/Total':>16}  {'AvgF_forget':>12}  {'AvgF_retain':>12}  {'Params':>6}")
        logger.info(f"  {'-'*28}  {'-'*8}  {'-'*16}  {'-'*12}  {'-'*12}  {'-'*6}")

        sorted_groups = sorted(
            group_stats.items(),
            key=lambda x: x[1]["active"] / max(x[1]["total"], 1),
            reverse=True
        )

        for group_key, g in sorted_groups:
            if g["total"] == 0:
                continue
            location, layer_type = group_key.split("|")
            density   = g["active"] / g["total"]
            avg_ff    = g["ff"] / g["count"]
            avg_fr    = g["fr"] / g["count"]
            label     = f"{location}/{layer_type}"

            # flag high-signal groups
            flag = ""
            if density > 0.20:
                flag = " ◄ HIGH"
            elif avg_ff > avg_fr * 10:
                flag = " ◄ forget-dominant"

            logger.info(
                f"  {label:<28}  {density:>7.3f}  "
                f"{g['active']:>7,}/{g['total']:>7,}  "
                f"{avg_ff:>12.2e}  {avg_fr:>12.2e}  "
                f"[{g['count']:>3} params]{flag}"
            )

        logger.info("[Mask] ────────────────────────────────────────────────────────")

    # ── Step 6: rebuild per-parameter masks from flat mask ───────────────────
    masks = [None] * len(accum_fisher_f)
    offset = 0
    for i, f in enumerate(accum_fisher_f):
        if f is None:
            continue
        n = f.numel()
        masks[i] = mask_flat[offset:offset + n].reshape(f.shape)
        offset += n

    return masks, mask_flat


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API: compute mask (with optional EMA smoothing for recomputation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dual_importance_mask(
    model,
    forget_dl,
    remain_dl,
    parameters,
    param_names,
    descriptions,
    class_to_forget,
    beta,
    device,
    target_density=0.10,       # 10% is a safer default than 15% — sweep this
    lambda_tradeoff=1.0,
    importance_variant="both",
    previous_mask_flat=None,   # pass the previous mask for EMA smoothing
    ema_alpha=0.3,             # weight of new mask in EMA:  new = α·new + (1-α)·old
    logger=None,
    max_fisher_batches=80,           # limit batches for faster mask computation during tuning
):
    """
    Compute a parameter importance mask for MUNBa unlearning.

    Key design decisions vs. original
    ───────────────────────────────────
    1. GLOBAL normalisation — Fisher scores are concatenated across ALL layers
       before any normalisation.  Per-layer Z-scoring (original) destroys the
       signal that some layers matter more than others globally.

    2. Ensemble scoring — ratio score + difference score are combined.
       Using only the difference (original) ignores multiplicative dominance.

    3. No Gumbel noise — the mask is deterministic.  Gumbel noise is designed
       for differentiable sampling during training, not for a one-shot mask.
       It adds uncontrolled variance with no benefit when the mask is fixed.

    4. EMA recomputation support — pass previous_mask_flat to smooth the
       transition between mask versions across epochs, preventing sudden
       mask thrashing.

    Args:
        target_density   : fraction of parameters to activate. Tune empirically
                           using the forget/retain FID tradeoff curve.
        previous_mask_flat: if recomputing each epoch, pass the previous flat
                           mask to apply EMA smoothing.
        ema_alpha        : EMA weight for new mask (0 = never update, 1 = no EMA)
    """
    if logger:
        logger.info(
            f"[Mask] Computing — density={target_density}  λ={lambda_tradeoff}  "
            f"ema={'yes' if previous_mask_flat is not None else 'no (first compute)'}"
        )

    accum_fisher_f, accum_fisher_r = _accumulate_fisher(
        model, forget_dl, remain_dl, parameters,
        descriptions, class_to_forget, beta, device,
        max_batches=max_fisher_batches
    )

    masks, mask_flat = _build_mask_from_fisher(
        parameters, accum_fisher_f, accum_fisher_r,
        target_density, lambda_tradeoff, importance_variant, logger, param_names=param_names
    )

    # ── EMA smoothing over mask recomputations ────────────────────────────────
    # When recomputing each epoch, a hard swap can destabilise training because
    # parameters that were receiving gradients suddenly lose them.  EMA smoothing
    # keeps the mask mostly stable while letting important new parameters in
    # gradually.
    if previous_mask_flat is not None:
        if logger:
            logger.info(f"[Mask] Applying EMA smoothing (α={ema_alpha})")

        # Soft blend: treat bool masks as float probabilities
        soft_new  = mask_flat.float()
        soft_prev = previous_mask_flat.float()
        blended   = ema_alpha * soft_new + (1.0 - ema_alpha) * soft_prev

        # Re-threshold: keep exactly the same number of active parameters
        k = mask_flat.sum().item()
        top_indices    = torch.topk(blended, k).indices
        mask_flat_ema  = torch.zeros_like(mask_flat)
        mask_flat_ema[top_indices] = True

        # Rebuild per-layer masks from smoothed flat mask
        masks_ema = [None] * len(masks)
        offset = 0
        for i, m in enumerate(masks):
            if m is None:
                continue
            n = m.numel()
            masks_ema[i] = mask_flat_ema[offset:offset + n].reshape(m.shape)
            offset += n

        if logger:
            changed = (mask_flat_ema != mask_flat).sum().item()
            logger.info(f"[Mask] EMA changed {changed:,} parameter slots from raw mask")

        del accum_fisher_f, accum_fisher_r
        torch.cuda.empty_cache()
        return masks_ema, mask_flat_ema

    del accum_fisher_f, accum_fisher_r
    torch.cuda.empty_cache()
    return masks, mask_flat
