import argparse
import json
import os
import random
import time

import numpy as np
import torch
from Bio import SeqIO
from obonet import read_obo
import pandas as pd
import nltk


def prepare_data(directory):
    go_ontology = read_obo("https://current.geneontology.org/ontology/go.obo")
    uniprot_kb = SeqIO.parse("./data/uniprot_sprot.fasta", "fasta")
    colnames = ["db", "db_obj_id", "db_symbol", "qualifier", "go_id", "db_ref", "evidence_code",
                "with_from", "aspect", "db_obj_name", "db_obj_synonym", "db_obj_type", "taxon", "date",
                "assigned_by", "annotation_ext", "gene_prod_form_id"]
    annotations = pd.read_csv("./data/goa_human.gaf", sep="\t", comment="!", names=colnames)
    annotations = annotations.loc[annotations["db"] == "UniProtKB", ["db_obj_id", "go_id"]].drop_duplicates()

    protein_sequences = {"id": [], "sequence": []}
    for entry in uniprot_kb:
        protein_sequences["id"].append(entry.name.split("|")[1])
        protein_sequences["sequence"].append(str(entry.seq))

    protein_sequences = pd.DataFrame(protein_sequences).set_index("id")
    protein_sequences = protein_sequences[protein_sequences.index.isin(annotations["db_obj_id"].to_list())]

    go_nodes = {"go_id": [], "name": [], "description": []}
    for name, node in go_ontology.nodes(data=True):
        go_nodes["go_id"].append(name)
        go_nodes["name"].append(node["name"])
        go_nodes["description"].append(node["def"].split('"')[1])

    go_nodes = pd.DataFrame.from_dict(go_nodes).set_index("go_id")
    go_nodes["id"] = np.arange(go_nodes.shape[0])
    protein_sequences["id"] = np.arange(protein_sequences.shape[0])

    annotations = annotations.loc[(annotations["db_obj_id"].isin(protein_sequences.index)) &
                                  (annotations["go_id"].isin(go_nodes.index)), :]

    annotations["protein_id"] = protein_sequences.loc[annotations["db_obj_id"], "id"].values
    annotations["function_id"] = go_nodes.loc[annotations["go_id"], "id"].values

    go_nodes["function_tokens"] = (go_nodes["name"] + " " + go_nodes["description"]).apply(tokenize_function)
    protein_sequences["sequence_tokens"] = protein_sequences["sequence"].apply(tokenize_protein)

    enc_protein = TokenEncoder()
    enc_protein.fit(protein_sequences["sequence_tokens"].to_list())
    enc_function = TokenEncoder()
    enc_function.fit(go_nodes["function_tokens"].to_list())

    protein_tensor = [enc_protein.encode(prot, 300) for prot in protein_sequences["sequence_tokens"].to_list()]
    function_tensor = [enc_function.encode(fun, 300) for fun in go_nodes["function_tokens"].to_list()]

    os.makedirs(f"./data/{directory}", exist_ok=True)

    torch.save(torch.stack(protein_tensor, 0), f"./data/{directory}/protein.pt")
    torch.save(torch.stack(function_tensor, 0), f"./data/{directory}/function.pt")
    annotations.loc[:, ["protein_id", "function_id"]].to_csv(f"./data/{directory}/relations.csv")

    enc_protein.save(f"./data/{directory}/enc_protein")
    enc_function.save(f"./data/{directory}/enc_function")


class TokenEncoder:
    def __init__(self):
        self.vocab = {}

    def save(self, path):
        with open(path, "w+") as file:
            json.dump(self.vocab, file)

    def read(self, path):
        with open(path, "r") as file:
            self.vocab = json.load(file)
        return self

    def fit(self, data):
        for entry in data:
            for token in entry:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, tokens, length):
        result = torch.ones([length]) * len(self.vocab)
        result[0] = len(self.vocab) + 1
        for i, token in enumerate(tokens, 1):
            if i >= length:
                break
            result[i] = self.vocab[token]
        return result


def tokenize_protein(protein_seq):
    return list(protein_seq)


def tokenize_function(function_name_desc):
    return nltk.word_tokenize(function_name_desc)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--expdir", default="exp1")
    args = args.parse_args()
    t = time.time()
    prepare_data(args.expdir)
    print(f"Preparation of training data took {time.time() - t}s")
