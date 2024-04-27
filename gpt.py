import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, num_embeddings, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        # store trill as parameters but not involve it during training
        self.register_buffer("trill", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        bsz, seq_len, channel = x.shape
        q = self.query(x)  # (bsz,seq_len,head_size)
        k = self.key(x)  # (bsz,seq_len,head_size)
        v = self.value(x)  # (bsz,seq_len,head_size)
        # compute attention score
        attn = q @ k.transpose(
            -2, -1
        )  # (bsz,seq_len,head_size) @ (bsz,head_size,seq_len) -> (bsz,seq_len,seq_len)
        attn = attn * 1.0 / math.sqrt(k.size(-1))  # scale by 1/sqrt(d)
        # perform causal masking to prevent model "see" future tokens
        masked_attn = attn.masked_fill(
            self.trill[:seq_len, :seq_len] == 0, float("-inf")
        )  # (bsz,seq_len,seq_len)
        # compute softmax
        out_attn = F.softmax(masked_attn, dim=-1)  # (bsz,seq_len,seq_len)
        # get information out from value
        out = out_attn @ v  # (bsz,seq_len,seq_len) @ (bsz,seq_len,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    run multiple self-attention in parallel
    """

    def __init__(self, num_embeddings, num_heads, head_size, block_size):
        super().__init__()
        self.m_heads = nn.ModuleList(
            SelfAttention(num_embeddings, head_size, block_size)
            for n in range(num_heads)
        )
        self.proj = nn.Linear(head_size * num_heads, num_embeddings)

    def forward(self, x):
        out = torch.cat([mh(x) for mh in self.m_heads], dim=-1)
        out = self.proj(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, num_embeddings, hs_scale_factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, hs_scale_factor * num_embeddings),
            nn.ReLU(),
            nn.Linear(hs_scale_factor * num_embeddings, num_embeddings),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, num_embeddings, num_heads, head_size, block_size, hs_scale_factor
    ):
        super().__init__()
        self.head = MultiHeadAttention(num_embeddings, num_heads, head_size, block_size)
        self.ffw = FeedFoward(num_embeddings, hs_scale_factor)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        # here are slightly different than original paper
        # we apply layer norm before attention & before ffw
        out = self.head(self.ln1(x))
        out = self.ffw(self.ln2(out))
        return out


class GPT(nn.Module):
    def __init__(
        self,
        num_layer,
        vocab_size,
        num_embeddings,
        num_heads,
        head_size,
        block_size,
        hs_scale_factor,
    ):
        """
        hs_scale_factor: hidden states scale factor
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, num_embeddings)
        # simple position embedding
        self.position_embeddings = nn.Embedding(block_size, num_embeddings)
        self.block = nn.Sequential(
            *[
                TransformerBlock(
                    num_embeddings, num_heads, head_size, block_size, hs_scale_factor
                )
                for n in range(num_layer)
            ]
        )
        self.final_ln = nn.LayerNorm(num_embeddings)  # final layer norm
        self.lm_head = nn.Linear(num_embeddings, vocab_size)

    @property
    def num_parameters(self):
        # Calculate the total number of parameters in all components of the GPT model
        params = (
            sum(p.numel() for p in self.parameters()) / 1e6
        )  # use self.parameters()
        return f"{params:.2f}M parameters"  # Formatted to show up to two decimal places

    def forward(self, x):
        bsz, seq_len = x.shape
        out = self.token_embeddings(x)
        pos_emb = self.position_embeddings(torch.arange(seq_len, device=x.device))
        out = out + pos_emb
        out = self.block(out)
        out = self.final_ln(out)
        logits = self.lm_head(out)

        return logits

    @torch.no_grad()
    def generate(self, prompt, tokenizer, vocab, max_new_tokens, device):
        self.eval()
        tokens = [vocab[token] for token in tokenizer(prompt)]
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            output = self(tokens)
            probs = F.softmax(output[0, -1], dim=0)
            next_token = torch.multinomial(probs, 1).item()
            tokens = torch.cat([tokens, torch.tensor([[next_token]]).to(device)], dim=1)
        generated_text = " ".join(
            vocab.get_itos()[token] for token in tokens[0].cpu().numpy()
        )
        return generated_text
