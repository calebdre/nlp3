import os
import torch
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from cnn import CNN
from gru import GRU
from analyzer import Analyzer

class Runner:
    def __init__(self, data_train, data_val, use_gpu=False):
        self.data_val = data_val
        self.data_train = data_train
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.analyzer = Analyzer(False)
            
        if self.use_gpu:
            print("Using GPU\n")
        
    def train(
        self, epochs, learning_rate, kernel_size, hidden_size, 
        model_cls, interaction, dropout
    ):
        if model_cls == "cnn":
            model = CNN(
                embedding = self.data_train.vocab_embedding, 
                embedding_size = self.data_train.vocab_embedding_size, 
                lengths = self.data_train.lengths(),
                kernel_size = kernel_size, 
                hidden_size = hidden_size,
                interaction = interaction,
                dropout = dropout
            )
        else:
            model = GRU(
                embedding = self.data_train.vocab_embedding,
                embedding_size = self.data_train.vocab_embedding_size, 
                encoding_size = hidden_size,
                interaction = interaction,
                dropout = dropout
            )
        if self.use_gpu:
            model = model.cuda()

        loader = self.data_train.get_loader()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        losses = []
        accuracies = []
        for epoch in range(1, epochs+1):
            e_loss = []
            print("\nStarting epoch {}".format(epoch))
            for i, (s1, s2, labels) in enumerate(loader):
                if self.use_gpu:
                    s1, s2, labels = s1.cuda(), s2.cuda(), labels.cuda()
                model.train()
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(s1, s2)
                instance_loss = loss_fn(logits, labels)

                # Backward and optimize
                instance_loss.backward()
                optimizer.step()
                
                losses.append(instance_loss.item())
                e_loss.append(instance_loss.item())
                
                # validate every 100 iterations
                if i > 0 and i % 100 == 0:
                    val_acc = self.validate(model)
                    accuracies.append(val_acc)
                    print('Epoch: [{}/{}]\tStep: [{}/{}]\tValidation Acc: {:.4f}'.format(
                            epoch, epochs, i, len(loader), val_acc
                        )
                    )
#             self.analyzer.plot_live_lr(e_loss, title="Epoch {}".format(epoch))

        avg_acc = sum(accuracies[-5:]) / 5
        
        self.analyzer.record(model.cpu(), losses, epochs=epochs, accuracies=accuracies, learning_rate=learning_rate, hidden_size=hidden_size, kernel_size=kernel_size,validation_accuracy=avg_acc, model_name=model_cls, dropout = dropout, interaction=interaction, data_length=32 * len(loader))
        self.analyzer.print_validation_results(self, model_cls, model)
        print("Final Accuracy: {}".format(avg_acc))

    def validate(self, *models, data = None, keep_predictions = False):
        total = torch.Tensor([0])
        
        corrects = [torch.Tensor([0]) for m in models]
        models = [model.eval() for model in models]
        all_labels = torch.Tensor([]).long()
        all_preds = [torch.Tensor([]).long() for m in models]
        all_s1s = []
        all_s2s = []
        genres = []
        
        if self.use_gpu:
            total = total.cuda()
            
            models = [model.cuda() for model in models]
            corrects = [correct.cuda() for correct in corrects]
            all_preds = [pred.cuda() for pred in all_preds]
            all_labels = all_labels.cuda()
        
        if data is None:
            data = self.data_val
            
        loader = data.get_loader()
        llen = len(loader)
        for i, batch in enumerate(loader):
            if "mnli" in data.mode:
                s1, s2, labels, genre = batch
            else:
                s1, s2, labels = batch
                
            if self.use_gpu:
                s1, s2, labels = s1.cuda(), s2.cuda(), labels.cuda()
            for j, model in enumerate(models):
                logits = model(s1, s2)
                preds = logits.max(1)[1]
                all_preds[j] = torch.cat((all_preds[j], preds))
                corrects[j] += preds.eq(labels).sum().item()
            
            total += labels.size(0)
            all_s1s += [s for s in s1]
            all_s2s += [s for s in s2]
            if "mnli" in data.mode:
                genres += genre
            all_labels = torch.cat((all_labels, labels))

        print(corrects, total)
        accuracies = [(100 * correct / total).item() for correct in corrects]
        
        if len(models) == 1:
            all_preds = all_preds[0]
            accuracies = accuracies[0]
        
        if keep_predictions:
            output = [accuracies, all_s1s, all_s2s, all_labels, all_preds]
            if "mnli" in data.mode:
                output.append(genres)
            
            return output
        else:
            return accuracies