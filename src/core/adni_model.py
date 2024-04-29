import torch
import torch.nn as nn

from ffn_encoder import FFNEncoder
from gnn import MML


class ADNIBackbone(nn.Module):
    def __init__(
            self,
            embedding_size,
            dropout,
            ffn_layers,
            gnn_layers,
            gnn_norm=None,
            device="cpu",
    ):
        super(ADNIBackbone, self).__init__()

        self.embedding_size = embedding_size
        self.dropout = dropout
        self.ffn_layers = ffn_layers
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)

        self.x1_encoder = FFNEncoder(input_dim=228,
                                     hidden_dim=embedding_size,
                                     output_dim=embedding_size,
                                     num_layers=ffn_layers,
                                     dropout_prob=dropout,
                                     device=device)
        self.x1_mapper = nn.Linear(embedding_size, embedding_size)

        self.x2_encoder = FFNEncoder(input_dim=227,
                                     hidden_dim=embedding_size,
                                     output_dim=embedding_size,
                                     num_layers=ffn_layers,
                                     dropout_prob=dropout,
                                     device=device)
        self.x2_mapper = nn.Linear(embedding_size, embedding_size)

        self.x3_encoder = FFNEncoder(input_dim=620,
                                     hidden_dim=embedding_size,
                                     output_dim=embedding_size,
                                     num_layers=ffn_layers,
                                     dropout_prob=dropout,
                                     device=device)
        self.x3_mapper = nn.Linear(embedding_size, embedding_size)

        self.mml = MML(num_modalities=3,
                       hidden_channels=embedding_size,
                       num_layers=gnn_layers,
                       dropout=dropout,
                       normalize_embs=gnn_norm,
                       num_classes=3)

    def forward(
            self,
            x1,
            x1_flag,
            x2,
            x2_flag,
            x3,
            x3_flag,
            label,
            label_flag,
            **kwargs,
    ):
        x1_flag = x1_flag.to(self.device)
        x2_flag = x2_flag.to(self.device)
        x3_flag = x3_flag.to(self.device)
        label = label.to(self.device)
        label_flag = label_flag.to(self.device)

        x1_embedding = self.x1_encoder(x1)
        x1_embedding = self.x1_mapper(x1_embedding)
        x1_embedding[x1_flag == 0] = 0
        x1_embedding = self.dropout_layer(x1_embedding)

        x2_embedding = self.x2_encoder(x2)
        x2_embedding = self.x2_mapper(x2_embedding)
        x2_embedding[x2_flag == 0] = 0
        x2_embedding = self.dropout_layer(x2_embedding)

        x3_embedding = self.x3_encoder(x3)
        x3_embedding = self.x3_mapper(x3_embedding)
        x3_embedding[x3_flag == 0] = 0
        x3_embedding = self.dropout_layer(x3_embedding)

        # gnn
        loss = self.mml(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            x3_embedding, x3_flag,
            label, label_flag,
        )
        return loss

    def inference(
            self,
            x1,
            x1_flag,
            x2,
            x2_flag,
            x3,
            x3_flag,
            **kwargs,
    ):
        x1_flag = x1_flag.to(self.device)
        x2_flag = x2_flag.to(self.device)
        x3_flag = x3_flag.to(self.device)

        x1_embedding = self.x1_encoder(x1)
        x1_embedding = self.x1_mapper(x1_embedding)
        x1_embedding[x1_flag == 0] = 0

        x2_embedding = self.x2_encoder(x2)
        x2_embedding = self.x2_mapper(x2_embedding)
        x2_embedding[x2_flag == 0] = 0

        x3_embedding = self.x3_encoder(x3)
        x3_embedding = self.x3_mapper(x3_embedding)
        x3_embedding[x3_flag == 0] = 0

        # gnn
        y_scores, logits = self.mml.inference(
            x1_embedding, x1_flag,
            x2_embedding, x2_flag,
            x3_embedding, x3_flag,
        )
        return y_scores, logits


if __name__ == "__main__":
    from src.dataset.adni_dataset import ADNIDataset
    from torch.utils.data import DataLoader

    dataset = ADNIDataset(split="train", load_no_label=True)
    data_loader = DataLoader(dataset, batch_size=8)
    batch = next(iter(data_loader))

    model = ADNIBackbone(
        embedding_size=128,
        dropout=0.1,
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
