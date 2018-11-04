import os
import torch
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from cnn import CNN
from gru import GRU

models_dir = "./"

class Runner:
    summaries = []
    
    def __init__(self, data_train, data_val, use_gpu=False):
        self.data_val = data_val
        self.data_train = data_train
        self.use_gpu = use_gpu and torch.cuda.is_available()
            
        if self.use_gpu:
            print("Using GPU\n")
        
    def train(
        self, epochs, learning_rate, kernel_size, hidden_size, 
        model_cls, interaction, hypothesis, dropout
    ):
        if model_cls == "cnn":
            model = CNN(
                embedding = self.data_train.vocab_embedding, 
                embedding_size = self.data_train.vocab_embedding_size, 
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
                
                # validate every 100 iterations
                if i > 0 and i % 50 == 0:
                    val_acc = self.validate(model)
                    accuracies.append(val_acc)
                    print('Epoch: [{}/{}]\tStep: [{}/{}]\tValidation Acc: {:4f}'.format(
                            epoch, epochs, i, len(loader), val_acc
                        )
                    )

        #self.plot_learning_rate(losses)
        avg_acc = sum(accuracies[-5:]) / 5
        print("Recording...\n")
        self.record(model.cpu(), losses, epochs=epochs, learning_rate=learning_rate, hidden_size=hidden_size, kernel_size=kernel_size,validation_accuracy=avg_acc, model_name=model_cls, dropout = dropout, interaction=interaction, hypothesis=hypothesis, data_length=32 * len(loader))
    
    def plot_learning_rate(self, losses):
        plt.plot(losses)
        plt.xlabel("Time-Step")
        plt.ylabel("Loss")
        plt.title("Run #{} Learning Rate".format(len(self.summaries) + 1))
        plt.show()
    
    def record(self, model, losses, **kwargs):
        run_info = {
            "losses": losses, 
            "model": model,
            "run_data": kwargs
        }
        self.summaries.append(run_info)
        
        with open("{}/run-{}.pkl".format(models_dir, int(time.time())), "wb+") as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)

    def validate(self, model):
        model.eval()
        correct = torch.Tensor([0])
        total = torch.Tensor([0])
        
        if self.use_gpu:
            model = model.cuda()
            correct = correct.cuda()
            total = total.cuda()
            
        loader = self.data_val.get_loader()
        llen = len(loader)
        for i, (s1, s2, labels) in enumerate(loader):
            if self.use_gpu:
                s1, s2, labels = s1.cuda(), s2.cuda(), labels.cuda()
            logits = model(s1, s2)
            predictions = logits.max(1)[1]
            total += labels.size(0)
            correct += predictions.eq(labels.view_as(predictions)).sum().item()
        return (100 * correct / total).item()

    @staticmethod
    def summarize():
        print("hydrating runs...")
        summaries = []
        for fname in os.listdir(models_dir):
            if ".pkl" in fname:
                with open("{}/{}".format(models_dir, fname), "rb") as f:
                    summaries.append(pickle.load(f)["run_data"])
        
        print("Summary:\n")
        pretty = tabulate(
            summaries, 
            headers="keys", 
            tablefmt='github', 
            showindex=range(1, len(summaries)+1)
        )
        print(pretty)
            
    def test(self):
        raise Exception("Not implemented")

if __name__ == "__main__":
    Runner.summarize()