import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import set_seed, processed_data_path, load_pickle

# set random seed
set_seed(42)

# set cuda devices
device = "cuda:0"
# device = "cpu"

# set inference hyper-parameters
batch_size = 1024
num_workers = 8


class CodeData(Dataset):
    def __init__(self, dataset):
        self.vocab_dir = os.path.join(processed_data_path, f"{dataset}/vocab.pkl")
        self.vocabs = load_pickle(self.vocab_dir)
        self.vocabs_size = len(self.vocabs)

    def __getitem__(self, index):
        return self.vocabs.idx2word[index]

    def __len__(self):
        return self.vocabs_size


def get_code_embeddings(loader, model, tokenizer):
    all_embeddings = []
    for i, text in tqdm(enumerate(loader)):
        with torch.no_grad():
            text_tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tokenized = text_tokenized.to(device)
            embeddings = model(**text_tokenized).last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def save_embeddings_to_file(embeddings, file_name):
    with open(file_name, "w") as f:
        count, dim = embeddings.shape
        f.write(f"{count} {dim}\n")
        for embedding in embeddings:
            f.write(f"{' '.join([str(i) for i in embedding.tolist()])}\n")
    return


def main():
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model.eval()
    model.to(device)

    data = CodeData(dataset="mimic4")
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings = get_code_embeddings(loader, model, tokenizer)
    save_embeddings_to_file(embeddings, os.path.join(processed_data_path, "mimic4/embeddings.txt"))

    data = CodeData(dataset="eicu")
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embeddings = get_code_embeddings(loader, model, tokenizer)
    save_embeddings_to_file(embeddings, os.path.join(processed_data_path, "eicu/embeddings.txt"))


if __name__ == '__main__':
    main()
