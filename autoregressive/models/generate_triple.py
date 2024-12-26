# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


@torch.no_grad()
def generate_fq(model, cond, max_new_tokens, cfg_scale=1.0, cfg_interval=-1, temperature=1.0, top_k=0, top_p=1.0, sample_logits=True):
    model.training=False
    cond = cond.unsqueeze(-1)
    if cfg_scale[0] > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    index_sample = sample_tri_fq(cond_combined, model, steps=max_new_tokens,
                                 sample_logits=sample_logits, top_k=top_k,
                                 temperature=temperature, top_p=top_p,
                                 cfg_scale=cfg_scale)
    return index_sample


@torch.no_grad()
def sample_tri_fq(x, model, steps, temperature=1.0, sample_logits=True,
           top_k=None, top_p=None, cfg_scale=1.0):
    k = 3
    bs, _ = x.shape
    device = x.device
    if cfg_scale[0] > 1.0:
        cond_token, uncond_token = torch.split(x, bs // 2, dim=0)
        sample_pre, sample_mid, sample_post = cond_token, cond_token, cond_token
    else:
        cond_token = x
        sample_pre, sample_mid, sample_post = cond_token, cond_token, cond_token
    
    cond_len = x.shape[1]
    if cfg_scale[0] > 1.0:
        max_batch_size = x.shape[0] // 2
    else:
        max_batch_size = x.shape[0]

    max_seq_length = cond_len + steps
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale[0] >= 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings_1.weight.dtype),
    
    for n in range(steps): ## start to sample
        if n == 0: #prefill operation
            input_pos = torch.arange(0, cond_len, device=device) #C
        elif n == 1:
            input_pos = torch.tensor([cond_len], device=device)
        else:
            input_pos = input_pos + 1
        
        h = model.generate_context(x, input_pos=input_pos, first_step=(n==0))
        
        x = []
        with torch.device(device): ##setup factorization 
            max_batch_size_cfg = max_batch_size * 2 if cfg_scale[0] >= 1.0 else max_batch_size
            model.setup_factorized_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings_1.weight.dtype)
        for i in range(k):
            if i == 0:## only CLS Token
                if cfg_scale[i] > 1.0:
                    factor_x = torch.concat([cond_token, uncond_token])
                else:
                    factor_x = cond_token
            factor_input_pos = torch.tensor([i], device=device)
            logits = model.decode_subtoken(h, factor_x, factor_input_pos, step=i, first_step=(i==0))

            if cfg_scale[i] > 1.0: #0611
                cond_logits, uncond_logits = torch.split(logits, bs // 2, dim=0)
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale[i]

            factor_x = sample_from_logits(logits, temperature[i], top_k[i], top_p[i])

            if i == 0:
                sample_pre = torch.cat((sample_pre, factor_x), dim=1)
            elif i==1:
                sample_mid = torch.cat((sample_mid, factor_x), dim=1)
            elif i==2:
                sample_post = torch.cat((sample_post, factor_x), dim=1)
            
            if cfg_scale[i] > 1.0:
                cfg_x = torch.concat([factor_x, factor_x])
                factor_x = [cfg_x, torch.concat([cond_token, uncond_token])]
                x.append(cfg_x)
            else:
                non_cfg_x = factor_x
                factor_x = (non_cfg_x, cond_token)
                x.append(non_cfg_x)
        
        if cfg_scale[0] > 1.0:
            x.append(torch.concat([cond_token, uncond_token]))
        else:
            x.append(cond_token)
    
    sample_pre = sample_pre[:, cond_len:]
    sample_mid = sample_mid[:, cond_len:]
    sample_post = sample_post[:, cond_len:]
    sample = (sample_pre, sample_mid, sample_post)

    return sample  


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits[:, -1, :] / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x
