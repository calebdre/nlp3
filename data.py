import sys
import pickle
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from load_word_embeddings import read_word_embeddings

data_dir = "../data"
# data_dir = "../floyd_data"
data_use = "100k"

class Data(Dataset):
    modes = ["snli_train", "snli_val", "mnli_train", "mnli_val"]
    unknown_token = "<unknown>"
    pad_token = "<pad>"
    
    def __init__(self, mode = "snli_train", num_rows = None, hydrate = False, batch_size = 32):
        self.batch_size = batch_size
        self.num_rows = num_rows
        if mode not in self.modes:
            raise Exception("'{}' is not a valid mode".format(mode))
        self.mode = mode
        
        if hydrate:
            print("Hydrating...")
            self.hydrate()
        else:
            print("Fetching files...")
            self.fetch()

            print("Fetching word embeddings...")
            self.prep_vocab()
            
            X = self.data
            y = X.pop("label")
            if "mnli" in mode:
                self.genre = X.pop("genre")
        
            print("Processing data...")        
            self.preprocess(X, y)
            
        if "mnli" in mode:
            self.data = (*self.data, self.genre)

    @property
    def unknown_idx(self):
        return self.vocab[self.unknown_token]
    
    @property
    def pad_idx(self):
        return self.vocab[self.pad_token]
    
    def get_loader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=True
        )

    def lengths(self):
        lengths = []
        d = self.get_data()
        x1, x2 = d[0], d[1]
            
        for x1i, x2i in zip(x1, x2):
            lengths.append(len(x1i))
            lengths.append(len(x2i))
        return set(lengths)
    
    def fetch(self):
        filename = "{}/{}.tsv".format(data_dir, self.mode)
        print("fetching {}".format(filename))
        
        self.data = pd.read_csv(filename, delimiter="\t", index_col=False)
        
    def prep_vocab(self):
        self.vocab_embedding, self.vocab_size, self.vocab_vec_size = read_word_embeddings()
        
        self.vocab_embedding.pop(",")
        self.vocab_embedding[self.unknown_token] = torch.zeros(self.vocab_vec_size)
        self.vocab_embedding[self.pad_token] = torch.zeros(self.vocab_vec_size)
        
        words = list(self.vocab_embedding.keys())
        self.vocab = dict(zip(words, range(len(words))))
        self.vocab_idx = dict(zip(range(len(words)), words))
                
        self.vocab_embedding = torch.stack(list(self.vocab_embedding.values()))
        self.vocab_embedding_size = self.vocab_embedding.shape[1]
    
    def preprocess(self, X, y):
        if self.num_rows is not None:
            print("Only taking {} rows".format(self.num_rows))
            X = X.iloc[:self.num_rows]
            y = y[:self.num_rows]
        
        y = torch.tensor(y.astype('category').cat.codes.values.astype(np.int32))    
        X = self.to_indices(X)
        
        X_1, X_2 = X.values.T
        self.data = (X_1, X_2, y)
        self.save()
        
    def hydrate(self):
        fname = "{}/data-{}-{}.pickle".format(data_dir, self.mode, data_use)
        with open(fname, "rb") as f:
            data = pickle.load(f)
            for key in data.keys():
                setattr(self, key, data[key])
        
    def save(self):
        fname = "{}/data-{}-{}.pickle".format(data_dir, self.mode, data_use)
        with open(fname, "wb+") as f:
            to_save = {
                "data": self.data,
                "mode": self.mode,
                "vocab": self.vocab,
                "vocab_idx": self.vocab_idx,
                "vocab_embedding": self.vocab_embedding,
                "vocab_embedding_size": self.vocab_embedding_size
            }
            
            if "mnli" in self.mode:
                to_save["genre"] = self.genre

            pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Finished saving to {}".format(fname))
        
    def to_index(self, token):
        try:
            return self.vocab[token]
        except:
            return self.vocab[self.unknown_token]
    
    def to_token(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        
        return self.vocab_idx[idx]
    
    def to_sentence(self, idxs):
        return " ".join([self.to_token(idx) for idx in idxs if idx != self.pad_idx])
    
    def to_indices(self, X):
        def to_index_func(sentence):
            indices = [] 
            for token in sentence.split(" "):
                idx = self.to_index(token)
                indices.append(idx)
            return indices
        
        column_values_transformer = lambda column: column.apply(to_index_func)
        return X.apply(column_values_transformer, axis=1)        
    
    def get_data(self, key=None):
        if key is None:
            return self.data
        else:
            data = [
                self.data[0][key],
                self.data[1][key],
                self.data[2][key]
            ]
            if "mnli" in self.mode:
                data.append(self.genre)

            return data
        
    def collate(self, batch):
        """
        Customized function for DataLoader that dynamically pads the batch so that all
        data have the same length
        """
        s1_list = []
        s2_list = []
        label_list = []
        genres = []            
        
        max_sentence_len = max([max(len(datum[0]), len(datum[1])) for datum in batch])
        # padding
        for datum in batch:
            label_list.append(datum[2])
            if "mnli" in self.mode:
                genres.append(datum[3])
            
            padded_s1 = np.pad(
                datum[0], 
                (0, max_sentence_len - len(datum[0])), 
                "constant",
                constant_values=self.pad_idx
            )
            padded_s2 =  np.pad(
                datum[1], 
                (0, max_sentence_len - len(datum[1])), 
                "constant",
                constant_values=self.pad_idx
            )
            
            s1_list.append(padded_s1)
            s2_list.append(padded_s2)

        collated = [
            torch.LongTensor(s1_list),
            torch.LongTensor(s2_list),
            torch.LongTensor(label_list)
        ]
        if "mnli" in self.mode:
            collated.append(genres)
        
        return collated
        
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, key):
        return self.get_data(key)
    
if __name__ == "__main__":
    print("Generating data file...")
    if len(sys.argv) > 1:
        num_rows = int(sys.argv[1])
        Data(num_rows=num_rows, mode="snli_train")
        Data(num_rows=num_rows, mode="snli_val")
    else:
        Data(mode="snli_train")
        Data(mode="snli_val")
    print("Finished!")
