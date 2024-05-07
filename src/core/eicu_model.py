import torch
import torch.nn as nn

from src.encoder.code_encoder import CodeEncoder
from src.encoder.ffn_encoder import FFNEncoder
from src.encoder.rnn_encoder import RNNEncoder
from gnn import MML


class eICUBackbone(nn.Module):
    def __init__(
            self,
            tokenizer,
            embedding_size,
            code_pretrained_embedding,
            code_layers,
            code_heads,
            dropout,
            rnn_layers,
            rnn_type,
            rnn_bidirectional,
            ffn_layers,
            gnn_layers,
            gnn_norm=None,
            device="cpu",
    ):
        super(eICUBackbone, self).__init__()

        self.embedding_size = embedding_size
        self.code_pretrained_embedding = code_pretrained_embedding
        self.code_layers = code_layers
        self.code_heads = code_heads
        self.dropout = dropout
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.rnn_bidirectional = rnn_bidirectional
        self.ffn_layers = ffn_layers
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)

        self.code_encoder = CodeEncoder(tokenizer=tokenizer,
                                        embedding_size=embedding_size,
                                        pretrained_embedding=code_pretrained_embedding,
                                        dropout=dropout,
                                        layers=code_layers,
                                        heads=code_heads,
                                        device=device)
        self.code_mapper = nn.Linear(embedding_size, embedding_size)

        self.rnn_encoder = RNNEncoder(input_size=158,
                                      hidden_size=embedding_size,
                                      num_layers=rnn_layers,
                                      rnn_type=rnn_type,
                                      dropout=dropout,
                                      bidirectional=rnn_bidirectional,
                                      device=device)
        self.rnn_mapper = nn.Linear(embedding_size, embedding_size)

        self.ffn_encoder = FFNEncoder(input_dim=36,
                                      hidden_dim=embedding_size,
                                      output_dim=embedding_size,
                                      num_layers=ffn_layers,
                                      dropout_prob=dropout,
                                      device=device)
        self.ffn_mapper = nn.Linear(embedding_size, embedding_size)

        self.mml = MML(num_modalities=3,
                       hidden_channels=embedding_size,
                       num_layers=gnn_layers,
                       dropout=dropout,
                       normalize_embs=gnn_norm,
                       num_classes=1)

    def forward(
            self,
            age,
            gender,
            ethnicity,
            types,
            codes,
            codes_flag,
            labvectors,
            labvectors_flag,
            apacheapsvar,
            apacheapsvar_flag,
            label,
            label_flag,
            **kwargs,
    ):
        codes_flag = codes_flag.to(self.device)
        labvectors_flag = labvectors_flag.to(self.device)
        apacheapsvar_flag = apacheapsvar_flag.to(self.device)
        label = label.to(self.device)
        label_flag = label_flag.to(self.device)

        # Code
        code_embedding = self.code_encoder(codes, types, age, gender, ethnicity)
        code_embedding = self.code_mapper(code_embedding)
        code_embedding[codes_flag == 0] = 0
        code_embedding = self.dropout_layer(code_embedding)

        # Lab
        lab_embedding = self.rnn_encoder(labvectors)
        lab_embedding = self.rnn_mapper(lab_embedding)
        lab_embedding[labvectors_flag == 0] = 0
        lab_embedding = self.dropout_layer(lab_embedding)

        # apacheapsvar
        apacheapsvar_embedding = self.ffn_encoder(apacheapsvar)
        apacheapsvar_embedding = self.ffn_mapper(apacheapsvar_embedding)
        apacheapsvar_embedding[apacheapsvar_flag == 0] = 0
        apacheapsvar_embedding = self.dropout_layer(apacheapsvar_embedding)

        # gnn
        loss = self.mml(
            code_embedding, codes_flag,
            lab_embedding, labvectors_flag,
            apacheapsvar_embedding, apacheapsvar_flag,
            label, label_flag,
        )
        return loss

    def inference(
            self,
            age,
            gender,
            ethnicity,
            types,
            codes,
            codes_flag,
            labvectors,
            labvectors_flag,
            apacheapsvar,
            apacheapsvar_flag,
            **kwargs,
    ):
        codes_flag = codes_flag.to(self.device)
        labvectors_flag = labvectors_flag.to(self.device)
        apacheapsvar_flag = apacheapsvar_flag.to(self.device)

        # Code
        code_embedding = self.code_encoder(codes, types, age, gender, ethnicity)
        code_embedding = self.code_mapper(code_embedding)
        code_embedding[codes_flag == 0] = 0

        # Lab
        lab_embedding = self.rnn_encoder(labvectors)
        lab_embedding = self.rnn_mapper(lab_embedding)
        lab_embedding[labvectors_flag == 0] = 0

        # apacheapsvar
        apacheapsvar_embedding = self.ffn_encoder(apacheapsvar)
        apacheapsvar_embedding = self.ffn_mapper(apacheapsvar_embedding)
        apacheapsvar_embedding[apacheapsvar_flag == 0] = 0

        # gnn
        y_scores, logits = self.mml.inference(
            code_embedding, codes_flag,
            lab_embedding, labvectors_flag,
            apacheapsvar_embedding, apacheapsvar_flag,
        )
        return y_scores, logits


if __name__ == "__main__":
    from src.dataset.eicu_dataset import eICUDataset
    from src.dataset.utils import eicu_collate_fn
    from torch.utils.data import DataLoader

    dataset = eICUDataset(split="train", task="mortality")
    data_loader = DataLoader(dataset, batch_size=128, collate_fn=eicu_collate_fn)
    batch = next(iter(data_loader))

    model = eICUBackbone(
        tokenizer=dataset.tokenizer,
        embedding_size=128,
        code_pretrained_embedding=True,
        code_layers=2,
        code_heads=2,
        dropout=0.1,
        rnn_layers=1,
        rnn_type="GRU",
        rnn_bidirectional=True,
        ffn_layers=2,
        gnn_layers=2,
        gnn_norm=None,
        device="cuda",
    )
    model.to("cuda")

    print(model)
    with torch.autograd.detect_anomaly():
        o = model(**batch)
        print(o)
        o.backward()

    print(model)
    with torch.autograd.detect_anomaly():
        o1, o2 = model.inference(**batch)
        print(o1.shape, o2.shape)
        o1.sum().backward()
