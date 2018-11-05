from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pickle
import os
import time
from tabulate import tabulate
import operator

models_dir = "models"

class Analyzer:
    runs = []
    
    def __init__(self):
        print("fetching runs...")
        for fname in os.listdir(models_dir):
            if ".pkl" in fname:
                with open("{}/{}".format(models_dir, fname), "rb") as f:
                    self.runs.append(pickle.load(f))
        self.runs = sorted(self.runs, key=lambda x: x["run_data"]["validation_accuracy"], reverse=True)
    
    def summarize(self):
        print("Summary:\n")
        run_data = [run["run_data"] for run in self.runs]    
        pretty = tabulate(
            run_data, 
            headers="keys", 
            tablefmt="grid", 
            showindex=range(0, len(run_data))
        )
        print(pretty)
    
    def record(self, model, losses, **kwargs):
        print("\nRecording...\n")
        run_info = {
            "losses": losses, 
            "model": model,
            "run_data": kwargs
        }
        self.runs.append(run_info)
        
        with open("{}/run-{}.pkl".format(models_dir, int(time.time())), "wb+") as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def plot_lr(self, *run_nums):
        if len(run_nums) == 1:
            num = run_nums[0]
            title = "Run #{} Learning Rate".format(1 if num == -1 else num+1)
        else:
            strs = [str(i) for i in run_nums]
            title = "Runs {} Learning Rate".format(", ".join(strs))
        
        runs = [self.runs[i] for i in run_nums]
        for i, run in enumerate(runs):
            y = run["losses"][::70]
            y = savgol_filter(y, 41, 5)
            plt.plot(y, label="Hidden Size: {}".format(run["run_data"]["hidden_size"]))
        
        if len(run_nums) > 1:
            plt.legend()
            
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()
        

if __name__ == "__main__":
    Analyzer().summarize()