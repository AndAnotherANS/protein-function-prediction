import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from transformer_model import Transformer
from data_utils import TokenEncoder
from torcheval.metrics.functional import binary_f1_score


class ProteinDataset(Dataset):
    def __init__(self, relations, labels, directory):
        self.relations = relations.long()
        self.labels = labels.long()
        self.proteins = torch.load(f"./data/{directory}/protein.pt").long()
        self.functions = torch.load(f"./data/{directory}/function.pt").long()

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, item):
        protein, function = self.relations[item]
        return self.proteins[protein], self.functions[function], self.labels[item]


def prepare_positive_negative(positive, max_neg_id):
    relations_negative = positive.clone()
    relations_negative[:, 1] = torch.randint(0,
                                             max_neg_id,
                                             [relations_negative.shape[0]])

    relations = torch.cat([positive, relations_negative], 0).long()

    labels = torch.cat([torch.ones([positive.shape[0]]), torch.zeros([relations_negative.shape[0]])]).long()
    return relations, labels


def train():
    directory = "exp1"
    relations_positive = pd.read_csv(f"./data/{directory}/relations.csv").values[:, 1:]

    np.random.shuffle(relations_positive)
    relations_positive_train = torch.tensor(relations_positive[:5000, :])
    relations_positive_test = torch.tensor(relations_positive[5000:20000, :])

    rel_train, label_train = prepare_positive_negative(relations_positive_train, relations_positive_train[:, 1].max())
    rel_test, label_test = prepare_positive_negative(relations_positive_test, relations_positive[:, 1].max())

    dataset_train = ProteinDataset(rel_train, label_train, directory)
    dataset_test = ProteinDataset(rel_test, label_test, directory)

    dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=True)

    prot_max_token = len(TokenEncoder().read(f"./data/{directory}/enc_protein").vocab)
    fun_max_token = len(TokenEncoder().read(f"./data/{directory}/enc_function").vocab)

    model_prot = Transformer(prot_max_token, 512, 5).cuda()
    model_fun = Transformer(fun_max_token, 512, 5).cuda()

    classifier = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid()).cuda()

    optim = torch.optim.Adam([*model_prot.parameters()] + [*model_fun.parameters()] + [*classifier.parameters()],
                             lr=0.0001)
    for epoch in range(50):
        model_prot.train()
        model_fun.train()
        losses = []
        f1 = []
        for prot, fun, label in tqdm.tqdm(dataloader):
            prot, fun, label = prot.cuda(), fun.cuda(), label.cuda()
            prot_encoding = model_prot(prot)
            fun_encoding = model_fun(fun)
            encoding = torch.cat([prot_encoding, fun_encoding], -1)
            pred = classifier(encoding).squeeze()
            loss = F.binary_cross_entropy(pred, label.float())

            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            model_prot.eval()
            model_fun.eval()
            for prot, fun, label in tqdm.tqdm(dataloader_test):
                prot, fun, label = prot.cuda(), fun.cuda(), label.cuda()
                prot_encoding = model_prot(prot)
                fun_encoding = model_fun(fun)
                encoding = torch.cat([prot_encoding, fun_encoding], -1)
                pred = classifier(encoding).squeeze()
                losses.append(((pred > 0.5).long() == label).float().mean().item())
                f1.append(binary_f1_score(pred, label))

        print("acc:", sum(losses) / len(losses))
        print("f1:", sum(f1) / len(f1))


if __name__ == '__main__':
    train()
