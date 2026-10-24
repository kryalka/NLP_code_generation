import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x):
        return self.split_heads(self.k(x)), self.split_heads(self.v(x))

    def forward(self, x, context=None, mask=None, causal=False, cache=None, static_kv=None):
        q = self.split_heads(self.q(x))
        if static_kv is not None:
            k, v = static_kv
        else:
            k, v = self.project_kv(x if context is None else context)
            if cache is not None:
                k = torch.cat((cache[0], k), dim=2)
                v = torch.cat((cache[1], v), dim=2)

        result = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        result = result.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out(result), (k, v)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = Attention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        y, _ = self.self_attn(self.norm1(x), mask=src_mask)
        x = x + self.drop(y)
        return x + self.drop(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = Attention(d_model, num_heads, dropout)
        self.cross_attn = Attention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask):
        y, _ = self.self_attn(self.norm1(x), causal=True)
        x = x + self.drop(y)
        y, _ = self.cross_attn(self.norm2(x), context=memory, mask=src_mask)
        x = x + self.drop(y)
        return x + self.drop(self.ff(self.norm3(x)))

    def step(self, x, src_mask, self_cache, cross_kv):
        y, new_cache = self.self_attn(self.norm1(x), cache=self_cache)
        x = x + self.drop(y)
        y, _ = self.cross_attn(self.norm2(x), mask=src_mask, static_kv=cross_kv)
        x = x + self.drop(y)
        x = x + self.drop(self.ff(self.norm3(x)))
        return x, new_cache


class Transformer(nn.Module):
    def __init__(self, cfg, pad_id=0, eos_id=1):
        super().__init__()
        d = cfg["d_model"]
        self.pad_id, self.eos_id = pad_id, eos_id
        self.max_source_length = cfg["max_source_length"]
        self.max_target_length = cfg["max_target_length"]
      
        self.embedding = nn.Embedding(cfg["vocab_size"], d)
        self.src_pos = nn.Embedding(self.max_source_length, d)
        self.tgt_pos = nn.Embedding(self.max_target_length, d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.encoder = nn.ModuleList([
            EncoderLayer(d, cfg["num_heads"], cfg["d_ff"], cfg["dropout"])
            for _ in range(cfg["encoder_layers"])
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d, cfg["num_heads"], cfg["d_ff"], cfg["dropout"])
            for _ in range(cfg["decoder_layers"])
        ])
        self.enc_norm = nn.LayerNorm(d)
        self.dec_norm = nn.LayerNorm(d)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def source_mask(self, source):
        return (source != self.pad_id)[:, None, None, :]

    def encode(self, source):
        b, length = source.shape
        assert length <= self.max_source_length
        positions = torch.arange(length, device=source.device)[None, :]
        x = self.embedding(source)
        x = x + self.src_pos(positions)
        x = self.drop(x)
        mask = self.source_mask(source)
        for layer in self.encoder:
            x = layer(x, mask)
        return self.enc_norm(x), mask

    def forward(self, source, target_input, labels=None):
        memory, mask = self.encode(source)
        length = target_input.size(1)
        assert length <= self.max_target_length
        positions = torch.arange(length, device=target_input.device)[None, :]
        x = self.embedding(target_input)
        x = x + self.tgt_pos(positions)
        x = self.drop(x)
        for layer in self.decoder:
            x = layer(x, memory, mask)
        x = self.dec_norm(x)
        if labels is not None:
            x = x[labels.ne(-100)]
        return F.linear(x, self.embedding.weight)

    @torch.inference_mode()
    def generate(self, source, max_new_tokens=160, temperature=0.0, top_p=1.0,
                 num_candidates=1, return_scores=False, generator=None):
        memory, mask = self.encode(source)
        cross_kv = [layer.cross_attn.project_kv(memory) for layer in self.decoder]
        if num_candidates > 1:
            mask = mask.repeat_interleave(num_candidates, dim=0)
            cross_kv = [(k.repeat_interleave(num_candidates, dim=0),
                         v.repeat_interleave(num_candidates, dim=0)) for k, v in cross_kv]
        self_cache = [None] * len(self.decoder)
        batch_size = source.size(0) * num_candidates
        token = torch.full((batch_size, 1), self.pad_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
        outputs = []
        if return_scores:
            logprob = torch.zeros(batch_size, device=source.device)
            lengths = torch.zeros(batch_size, dtype=torch.long, device=source.device)

        for position in range(max_new_tokens):
            pos = torch.full_like(token, position)
            x = self.embedding(token)
            x = x + self.tgt_pos(pos)
            x = self.drop(x)
            for i, layer in enumerate(self.decoder):
                x, self_cache[i] = layer.step(x, mask, self_cache[i], cross_kv[i])
            logits = F.linear(self.dec_norm(x[:, -1]), self.embedding.weight)
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                probabilities = F.softmax(logits.float() / temperature, dim=-1)
                probabilities, indices = probabilities.sort(dim=-1, descending=True)
                remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
                probabilities.masked_fill_(remove, 0)
                sampled = torch.multinomial(probabilities, 1, generator=generator)
                next_token = indices.gather(1, sampled).squeeze(1)
            if return_scores:
                scores = F.log_softmax(logits.float(), dim=-1).gather(1, next_token[:, None]).squeeze(1)
                logprob += scores.masked_fill(finished, 0)
                lengths += ~finished
            next_token = torch.where(finished, self.pad_id, next_token)
            outputs.append(next_token)
            finished |= next_token.eq(self.eos_id)
            if bool(finished.all()):
                break
            token = next_token[:, None]

        generated = torch.stack(outputs, dim=1)
        if return_scores:
            return generated, logprob, lengths
        return generated

