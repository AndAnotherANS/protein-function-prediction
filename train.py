import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from transformer_model import Transformer
from data_utils import TokenEncoder
from torcheval.metrics.functional import binary_f1_score, binary_precision, binary_recall, binary_accuracy
from matplotlib import pyplot as plt

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
    num_epochs = 5
    step_size = 2
    relations_positive = pd.read_csv(f"./data/{directory}/relations.csv").values[:, 1:]

    train_proteins = np.random.choice(relations_positive[:, 0], [200], replace=False)
    relations_positive_train = torch.tensor(relations_positive[np.isin(relations_positive[:, 0], train_proteins)])
    relations_positive_val = torch.tensor(relations_positive[~np.isin(relations_positive[:, 0], train_proteins)][:10000])

    rel_train, label_train = prepare_positive_negative(relations_positive_train, relations_positive_train[:, 1].max())
    rel_val, label_val = prepare_positive_negative(relations_positive_val, relations_positive[:, 1].max())

    dataset_train = ProteinDataset(rel_train, label_train, directory)
    dataset_val = ProteinDataset(rel_val, label_val, directory)

    dataloader = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=True)

    prot_max_token = len(TokenEncoder().read(f"./data/{directory}/enc_protein").vocab)
    fun_max_token = len(TokenEncoder().read(f"./data/{directory}/enc_function").vocab)

    model_prot = Transformer(prot_max_token, 512, 5).cuda()
    model_fun = Transformer(fun_max_token, 512, 5).cuda()

    classifier = nn.CosineSimilarity(dim=-1)#nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid()).cuda()

    optim = torch.optim.Adam([*model_prot.parameters()] + [*model_fun.parameters()] + [*classifier.parameters()],
                             lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.5)
    best_loss = torch.inf
    loss_history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        model_prot.train()
        model_fun.train()
        losses = []

        for prot, fun, label in tqdm.tqdm(dataloader):
            prot, fun, label = prot.cuda(), fun.cuda(), label.cuda()
            prot_encoding = model_prot(prot)
            fun_encoding = model_fun(fun)
            #encoding = torch.cat([prot_encoding, fun_encoding], -1)
            pred = (classifier(prot_encoding, fun_encoding).squeeze() + 1.) / 2.
            loss = F.binary_cross_entropy(pred, label.float())

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

        scheduler.step()
        loss_val = evaluate(model_prot, model_fun, classifier, dataloader_val)
        loss_history["train"].append(sum(losses)/len(losses))
        loss_history["val"].append(loss_val)

        if loss_val < best_loss:
            best_models = {"protein": model_prot.state_dict(), "func": model_fun.state_dict(), "classifier": classifier.state_dict()}
            torch.save(best_models, f"./data/{directory}/best_model.pt")

    ax = plt.subplot(121)
    ax.set_title("train loss history")
    plt.plot(np.arange(len(loss_history["train"])), loss_history["train"])
    ax = plt.subplot(122)
    ax.set_title("validation loss history")
    plt.plot(np.arange(len(loss_history["val"])), loss_history["val"])
    plt.show()

def evaluate(model_prot, model_fun, classifier, dataloader_val):
    acc = []
    f1 = []
    precision = []
    recall = []
    losses = []
    with torch.no_grad():
        model_prot.eval()
        model_fun.eval()
        for prot, fun, label in tqdm.tqdm(dataloader_val):
            prot, fun, label = prot.cuda(), fun.cuda(), label.cuda()
            prot_encoding = model_prot(prot)
            fun_encoding = model_fun(fun)
            encoding = torch.cat([prot_encoding, fun_encoding], -1)
            pred = (classifier(prot_encoding, fun_encoding).squeeze() + 1.) / 2.
            acc.append(binary_accuracy(pred, label).item())
            f1.append(binary_f1_score(pred, label).item())
            precision.append(binary_precision(pred, label).item())
            recall.append(binary_recall(pred, label).item())
            losses.append(F.binary_cross_entropy(pred, label.float()).item())
    print("acc:", sum(acc) / len(acc))
    print("f1:", sum(f1) / len(f1))
    print("precision:", sum(precision) / len(precision))
    print("recall:", sum(recall) / len(recall))
    print("loss:", sum(losses) / len(losses))
    return sum(losses) / len(losses)

if __name__ == '__main__':
    train()
