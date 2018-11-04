from tabulate import tabulate
import matplotlib.pyplot as plt
import pickle
import os
import time

models_dir = "models"

class Analyzer:
    runs = []
    
    def __init__(self):
        print("fetching runs...")
        for fname in os.listdir(models_dir):
            if ".pkl" in fname:
                with open("{}/{}".format(models_dir, fname), "rb") as f:
                    self.runs.append(pickle.load(f))
    
    def summarize(self):
        print("Summary:\n")
        run_data = [run["run_data"] for run in self.runs]
        pretty = tabulate(
            run_data, 
            headers="keys", 
            tablefmt='github', 
            showindex=range(1, len(summaries)+1)
        )
        print(pretty)
    
    def record(self, model, losses, **kwargs):
        print("Recording...\n")
        run_info = {
            "losses": losses, 
            "model": model,
            "run_data": kwargs
        }
        self.runs.append(run_info)
        
        with open("{}/run-{}.pkl".format(models_dir, int(time.time())), "wb+") as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def plot_learning_rate(self, i = -1):
        title = "Run #{} Learning Rate".format(1 if i == -1 else i+1)
        plt.plot(self.runs[i]["losses"][::50])
        plt.xlabel("Time-Step")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()