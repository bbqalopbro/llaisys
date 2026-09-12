"""Single-request greedy driver, not a replacement serving scheduler.

Sampling and chunk policy stay in Python. Each call owns and finally releases
its request state; the caller owns any shared paged pool. No MTP is executed.
"""

import torch


@torch.inference_mode()
def greedy_generate(model, input_ids, *, max_new_tokens, eos_id, chunk_size=0,
                    cache_pool=None, cancelled=None, on_token=None, retain_logits=False, reuse_prefix=False):
    if type(reuse_prefix) is not bool or (reuse_prefix and (cache_pool is None or cache_pool.prefix is None)):
        raise ValueError("prefix reuse requires an explicitly enabled paged pool")
    for name, value, minimum in (("max_new_tokens", max_new_tokens, 1), ("chunk_size", chunk_size, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {name}")
    cfg = model.config
    if type(eos_id) is not int or not 0 <= eos_id < cfg.vocab_size:
        raise ValueError("invalid EOS ID")
    if not isinstance(input_ids, (list, tuple)) or not input_ids or any(
            type(i) is not int or not 0 <= i < cfg.vocab_size for i in input_ids):
        raise ValueError("input_ids must be nonempty vocabulary token IDs")
    capacity = cfg.max_seq_len - len(input_ids)
    if capacity <= 0:
        raise ValueError("prompt leaves no generation capacity")
    budget = min(max_new_tokens, capacity)
    tokens = torch.tensor([input_ids], dtype=torch.int64, device=model.embed.weight.device)
    generated, saved = [], []
    finish_reason, stop_reason = "length", "context_length" if capacity < max_new_tokens else "max_new_tokens"
    state = model.new_request(cache_pool=cache_pool)
    hit_tokens, published = 0, False
    try:
        if reuse_prefix:
            hit_tokens = state.attach_prefix(input_ids)
        size = chunk_size or tokens.shape[1]
        for position in range(state.position, tokens.shape[1], size):
            if cancelled is not None and cancelled():
                finish_reason, stop_reason = "cancelled", "cancelled"
                break
            final = position + size >= tokens.shape[1]
            output = model(tokens[:, position:position+size], state, emit_logits=final)
        else:
            if reuse_prefix:
                published = state.publish_prefix()
            for index in range(budget):
                if cancelled is not None and cancelled():
                    finish_reason, stop_reason = "cancelled", "cancelled"
                    break
                if index:
                    output = model(torch.tensor([[generated[-1]]], dtype=torch.int64, device=tokens.device), state)
                logits = output.logits
                if logits is None or logits.shape != (1, cfg.vocab_size) or not torch.isfinite(logits).all():
                    raise RuntimeError("generation requires finite [1, vocab_size] logits")
                token = logits.argmax(-1).item()
                generated.append(token)
                if retain_logits:
                    saved.append(logits.detach().float().cpu())
                if on_token is not None:
                    on_token(token)
                if token == eos_id:
                    finish_reason, stop_reason = "stop", "eos"
                    break
        cache_position = state.position
    finally:
        state.close()
    return {"input_tokens": len(input_ids), "generated_ids": generated, "output_tokens": len(generated),
            "max_new_tokens": max_new_tokens, "effective_token_budget": budget,
            "finish_reason": finish_reason, "stop_reason": stop_reason, "cache_position": cache_position,
            "prefill_chunk_size": chunk_size, "sampling": "greedy", "mtp": False,
            "prefix_hit_tokens": hit_tokens, "prefix_published": published,
            "request_closed": state.closed,
            **({"logits": torch.stack(saved) if saved else torch.empty((0, 1, cfg.vocab_size))}
               if retain_logits else {})}
