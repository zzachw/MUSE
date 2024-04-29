import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, bert_type="emilyalsentzer/Bio_ClinicalBERT", device="cpu") -> None:
        super().__init__()
        self.bert_type = bert_type
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.device = device

    def forward(self, text):
        text_tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        text_tokenized = text_tokenized.to(self.device)
        embeddings = self.model(**text_tokenized).pooler_output
        return embeddings


if __name__ == "__main__":
    from src.dataset.mimic4_dataset import MIMIC4Dataset
    from src.dataset.utils import mimic4_collate_fn
    from torch.utils.data import DataLoader

    dataset = MIMIC4Dataset(split="train", task="mortality")
    data_loader = DataLoader(dataset, batch_size=256, collate_fn=mimic4_collate_fn)
    batch = next(iter(data_loader))

    model = TextEncoder(device="cuda")
    model.to("cuda")
    print(model)
    with torch.autograd.detect_anomaly():
        o = model(batch["discharge"])
        print(o.shape)
        o.sum().backward()
