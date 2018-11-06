from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pickle
import os
import torch
from scipy.signal import lfilter
import time
from tabulate import tabulate
import operator

models_dir = "."

class Analyzer:
    runs = []
    
    def __init__(self, fetch=True):
        if fetch:
            print("fetching runs...")
            for fname in os.listdir(models_dir):
                if ".torch" in fname:
                    with open("{}/{}".format(models_dir, fname), "rb") as f:
                        m = torch.load(f, map_location=torch.device("cpu"))
                        m["run_data"]["filename"] = fname
                        if "hypothesis" in m["run_data"].keys():
                            m["run_data"].pop("hypothesis")
                        self.runs.append(m)
            self.runs = sorted(self.runs, key=lambda x: x["run_data"]["validation_accuracy"], reverse=True)
    
    def summarize(self):
        print("Summary:\n")
        run_data = [run["run_data"] for run in self.runs]    
        pretty = tabulate(
            run_data, 
            headers="keys", 
            tablefmt="grid",
            showindex="always"
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
        
        with open("{}/run-{}.torch".format(models_dir, int(time.time())), "wb+") as f:
            torch.save(run_info, f)
    
    def plot_live_lr(self, losses, title=None):
        t = int(len(losses)/3)
        if t % 2 == 0:
            t += 1
#         y = savgol_filter(losses, t, 4)
        y = lfilter(15, 1, y)
        plt.plot(y)
        plt.ylabel("Loss")
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_lr(self, *run_nums, zoom=None):
        if len(run_nums) == 1:
            num = run_nums[0]
            title = "Run #{} Learning Rate".format(1 if num == -1 else num+1)
        else:
            strs = [str(i) for i in run_nums]
            title = "Runs {} Learning Rate".format(", ".join(strs))
        
        runs = [self.runs[i] for i in run_nums]
        for i, run in enumerate(runs):
            y = run["losses"][::10]
            
            t = int(len(y)/2)
            if t % 2 == 0:
                t += 1
            y = savgol_filter(y, t, 3)
            if zoom is not None:
                y = y[zoom:]
            plt.plot(y, label="Run {} | Hidden Size: {}".format(run_nums[i], run["run_data"]["hidden_size"]))
        
        if len(run_nums) > 1:
            plt.legend()
            
        plt.xlabel("Time")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()
        

if __name__ == "__main__":
    Analyzer().summarize()