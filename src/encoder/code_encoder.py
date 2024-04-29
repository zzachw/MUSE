import math
import torch.nn as nn

from src.utils import *


class RelTemporalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000.0) / embedding_size))
        embedding = nn.Embedding(max_len, embedding_size)
        embedding.weight.data[:, 0::2] = torch.sin(position * div_term)
        embedding.weight.data[:, 1::2] = torch.cos(position * div_term)
        embedding.requires_grad = False
        self.embedding = embedding

    def forward(self, timestamps: torch.tensor):
        timestamps_emb = self.embedding(timestamps)
        return timestamps_emb


class Attention(nn.Module):
    def forward(self, query, key, value, mask, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        """
        :param query, key, value: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :return: [batch_size, seq_len, d_model]
        """

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask.unsqueeze(1), dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        mask = mask.sum(dim=-1) > 0
        x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Apply residual connection to any sublayer with the same size. """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Transformer Block = MultiHead Attention + Feed Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param dropout: dropout rate
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=4 * hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        print(f"TransformerBlock added with hid-{hidden}, head-{attn_heads}, in_hid-{2 * hidden}, drop-{dropout}")

    def forward(self, x, mask):
        """
        :param x: [batch_size, seq_len, hidden]
        :param mask: [batch_size, seq_len, seq_len]
        :return: batch_size, seq_len, hidden]
        """

        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return x


class CodeEncoder(nn.Module):
    def __init__(
            self,
            tokenizer,
            embedding_size: int,
            pretrained_embedding: bool,
            dropout: float,
            layers: int,
            heads: int,
            device="cpu",
    ):
        super(CodeEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self.dropout = dropout
        self.layers = layers
        self.heads = heads
        self.device = device

        # embedding
        if pretrained_embedding:
            mm = torch.Tensor(self.tokenizer.code_embeddings)
            assert self.tokenizer.code_vocabs_size == mm.shape[0]
            self.code_embedding = nn.Sequential(
                nn.Embedding(self.tokenizer.code_vocabs_size, mm.shape[1]),
                nn.Linear(mm.shape[1], embedding_size)
            )
            self.code_embedding[0].weight.data.copy_(mm)
            self.code_embedding[0].weight.requires_grad = False
        else:
            self.code_embedding = nn.Embedding(self.tokenizer.code_vocabs_size, embedding_size, padding_idx=0)
        self.type_embedding = nn.Embedding(self.tokenizer.type_vocabs_size, embedding_size, padding_idx=0)
        self.age_embedding = nn.Embedding(self.tokenizer.age_vocabs_size, embedding_size, padding_idx=0)
        self.gender_embedding = nn.Embedding(self.tokenizer.gender_vocabs_size, embedding_size, padding_idx=0)
        self.ethnicity_embedding = nn.Embedding(self.tokenizer.ethnicity_vocabs_size, embedding_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(dropout)

        # encoder
        self.transformer = nn.ModuleList([TransformerBlock(embedding_size, heads, dropout) for _ in range(layers)])

    def forward(
            self, codes, types, ages, genders, ethnicities
    ):
        codes = codes.to(self.device)
        types = types.to(self.device)
        ages = ages.to(self.device)
        genders = genders.to(self.device)
        ethnicities = ethnicities.to(self.device)

        mask = (types != 0)
        mask = torch.einsum("ab,ac->abc", mask, mask)

        """ embedding """
        # [# admissions, # batch_codes, embedding_size]
        codes_emb = self.code_embedding(codes)
        types_emb = self.type_embedding(types)
        ages_emb = self.age_embedding(ages)
        genders_emb = self.gender_embedding(genders)
        ethnicities_emb = self.ethnicity_embedding(ethnicities)
        demographics_emb = (ages_emb + genders_emb + ethnicities_emb).unsqueeze(1)
        emb = codes_emb + types_emb + demographics_emb
        emb[mask.sum(dim=-1) == 0] = 0
        emb = self.dropout_layer(emb)

        """ transformer """
        x = emb
        for transformer in self.transformer:
            x = transformer(x, mask)  # [# admissions, # batch_codes, embedding_size]

        cls_emb = x[:, 0, :]
        return cls_emb


if __name__ == "__main__":
    from src.dataset.mimic4_dataset import MIMIC4Dataset
    from src.dataset.utils import mimic4_collate_fn
    from torch.utils.data import DataLoader

    dataset = MIMIC4Dataset(split="train", task="mortality")
    data_loader = DataLoader(dataset, batch_size=256, collate_fn=mimic4_collate_fn)
    batch = next(iter(data_loader))

    model = CodeEncoder(tokenizer=dataset.tokenizer,
                        embedding_size=128,
                        pretrained_embedding=True,
                        dropout=0.1,
                        layers=3,
                        heads=8)
    print(model)
    with torch.autograd.detect_anomaly():
        o = model(batch["codes"],
                  batch["types"],
                  batch["age"],
                  batch["gender"],
                  batch["ethnicity"])
        print(o.shape)
        o.sum().backward()
