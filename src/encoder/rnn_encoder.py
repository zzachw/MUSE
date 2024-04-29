from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class RNNEncoder(nn.Module):
    """Recurrent neural network layer.

    This layer wraps the PyTorch RNN layer with masking and dropout support. It is
    used in the RNN model. But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        rnn_type: type of rnn, one of "RNN", "LSTM", "GRU". Default is "GRU".
        num_layers: number of recurrent layers. Default is 1.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            RNN layer. Default is 0.5.
        bidirectional: whether to use bidirectional recurrent layers. If True,
            a fully-connected layer is applied to the concatenation of the forward
            and backward hidden states to reduce the dimension to hidden_size.
            Default is False.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            rnn_type: str = "GRU",
            num_layers: int = 1,
            dropout: float = 0.5,
            bidirectional: bool = False,
            device: str = "cpu",
    ):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device = device

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
            self,
            x: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
        """
        # pytorch's rnn will only apply dropout between layers
        lengths = ((x != 0).sum(-1) > 0).sum(dim=-1)
        x = x.to(self.device)
        lengths[lengths == 0] = 1
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            # return outputs, last_outputs
            return last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            # outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            # outputs = self.down_projection(outputs)
            # return outputs, last_outputs
            return last_outputs


if __name__ == "__main__":
    from src.dataset.mimic4_dataset import MIMIC4Dataset
    from src.dataset.utils import mimic4_collate_fn
    from torch.utils.data import DataLoader

    dataset = MIMIC4Dataset(split="train", task="mortality")
    data_loader = DataLoader(dataset, batch_size=256, collate_fn=mimic4_collate_fn)
    batch = next(iter(data_loader))

    model = RNNEncoder(input_size=114,
                       hidden_size=128,
                       bidirectional=True,
                       device="cuda")
    model.to("cuda")
    print(model)
    with torch.autograd.detect_anomaly():
        o = model(batch["labvectors"])
        print(o.shape)
        o.sum().backward()
