from scipy.signal import savgol_filter
from termcolor import colored
import matplotlib.pyplot as plt
import pickle
import os
import torch
from scipy.signal import lfilter
import time
from tabulate import tabulate
import operator


class Analyzer:
    runs = []
    models_dir = "../models/old/old_best"
    
    def __init__(self, fetch=True, model_dir = None):
        if model_dir is not None:
            self.models_dir = model_dir
        if fetch:
            print("fetching runs...")
            for fname in os.listdir(self.models_dir):
                if ".torch" in fname:
                    with open("{}/{}".format(self.models_dir, fname), "rb") as f:
                        m = torch.load(f, map_location=torch.device("cpu"))
                        m["run_data"]["filename"] = fname
                        if "hypothesis" in m["run_data"].keys():
                            m["run_data"].pop("hypothesis")
                        if "accuracies" in m["run_data"]:
                            m["accuracies"] = m["run_data"]["accuracies"]
                            m["run_data"].pop("accuracies")
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

    def plot_lr(self, *run_nums, plot_accuracy=False, zoom=None, field = None, title_ = None, no_x = False, x_label=None, extra_labels= [], filename = None):
        if len(run_nums) == 1:
            num = run_nums[0]
            title = "Run #{} Learning Rate".format(1 if num == -1 else num+1)
        else:
            strs = [str(i) for i in run_nums]
            title = "Runs {} Learning Rate".format(", ".join(strs))
        
        if title_ is not None:
            title = title_
        
        runs = [self.runs[i] for i in run_nums]
        for i, run in enumerate(runs):
            if plot_accuracy == True:
                y = run["accuracies"]
            else:
                y = run["losses"]
            
            if field is not None:
                if type(field) == list:
                    y_label = ", ".join(["{}: {}".format(" ".join(f.split("_")).title(), run["run_data"][f]) for f in field])
                else:
                    y_label = "{}: {}".format(" ".join(field.split("_")).title(), run["run_data"][field])
            else:
                y_label = "Run {}".format(i)
            self.plot(y, y_label)
        
        if len(run_nums) > 1:
            plt.legend()
            
        if no_x == False:
            if x_label is not None:
                plt.xlabel(x_label)
            else:
                if plot_accuracy == True:
                    plt.xlabel("Epochs")
                    plt.ylabel("Accuracy")
                else:
                    plt.xlabel("Time")
                    plt.ylabel("Loss")
        else:
            plt.xticks([])
        
        for label in extra_labels:
            plt.plot([], [],'', label=label)
        
        plt.title(title)
        if filename is not None:
            plt.savefig(filename)
        plt.show()
    
    def plot(self, y, label = None):
        t = int(len(y)/3)
        if t % 2 == 0:
            t += 1
        y = savgol_filter(y, t, 4)
        if len(y) > 1000:
            y = y[300:-300]
        plt.plot(y, label=label)
        
    def print_validation_results(self, data, model_names, accs, s1_idxs, s2_idxs, labels, preds):
        label_map = {
            0: "entail",
            1: "contradict",
            2: "neutral"
        }

        color = lambda x: "green" if x else "red"
        marker = lambda x: "✔" if x else "ˣ"

        s1s = []
        s2s = []

        for s1, s2 in zip(s1_idxs, s2_idxs): 
            s1s.append(data.to_sentence(s1))
            s2s.append(data.to_sentence(s2))
        
        for i, (s1, s2, label, *model_preds) in enumerate(zip(s1s, s2s, labels, *preds)):
            model_texts = []
            for i, p in enumerate(model_preds):
                correct = p.eq(label).item() == 1
                mark = marker(correct)
                pred_text = label_map[p.item()]

                text = colored("[{}] {}: {}".format(mark, model_names[i], pred_text), color(correct))
                model_texts.append(text)

            label_text = colored("  Actual: {}".format(label_map[label.item()]), "blue")
            print("{}\nSentence 1:\n{}\n\nSetence 2:\n{}\n\n{}\n{}".format(i, s1, s2, label_text, "\n".join(model_texts)))
            print("\n------------------------------------------------------\n")
        

if __name__ == "__main__":
    Analyzer().summarize()