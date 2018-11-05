import sys
from time import time
import torch
from data import Data
from gru import GRU
from cnn import CNN
from runner import Runner
from analyzer import Analyzer

def run(params, d_train, d_val):
    runner = Runner(d_train, d_val, use_gpu = True)
    start = time()
    runner.train(**params)
    elapsed = time() - start
    print("\nTraining took {}s\n\n".format(elapsed))

def train(model_name, batch_size = 32, epochs = 10, hypothesis = None, altered_params=None):
    data_train = Data(mode ="snli_train", batch_size = batch_size, hydrate = True)
    data_val = Data(mode="snli_val", batch_size = batch_size, hydrate = True)
    
    params = {
        "hypothesis": hypothesis,
        "epochs": epochs,
        "learning_rate": .001,
        "kernel_size": 2,
        "hidden_size": 450,
        "model_cls": model_name,
        "interaction": "concat",
        "dropout": .6,
        "weight_decay": 0
    }

    if altered_params is not None:
        keys = list(altered_params.keys())
        vals = [altered_params[key] for key in keys]
        for ind_vals in zip(*vals):
            run_params = {**params}
            for i, val in enumerate(ind_vals):
                print("Setting {} to {}".format(keys[i], val))
                run_params[keys[i]] = val
            run(run_params, data_train, data_val)
    else:
        run(params, data_train, data_val)
    
    Analyzer().summarize()

if __name__ == "__main__":
    num_args = len(sys.argv)
    if num_args == 1:
        print("Please pass in a model: 'gru'|'cnn'")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name != "gru" and model_name != "cnn":
        print("Please pass in a model: 'gru'|'cnn'")
        sys.exit(1)


    if num_args == 3:
        train(model_name, epochs = int(sys.argv[2]))
    elif num_args == 4:
        print("Must pass in a [epochs] [hypothesis] [param] [values]")
    elif num_args > 4:

        epochs = int(sys.argv[2])
        hypothesis = sys.argv[3]
        
        param_names = ["weight_decay", "dropout", "learning_rate", "kernel_size", "encoder_size", "hidden_size", "model_cls", "interaction"]
        param_indices = [i for i, inp in enumerate(sys.argv) if inp in param_names]
        altered_params = {}
        for i in range(1, len(param_indices)+1):
            to = param_indices[i:] if i <= len(param_indices) else param_indices[i+1]
            if i == len(param_indices):
                idx = param_indices[i-1]
                prev = param_indices[i-2]
                altered_params[sys.argv[idx]] = [int(arg) if arg.isnumeric() else arg for arg in sys.argv[idx+1:]]
            else:
                idx = param_indices[i]
                prev = param_indices[i-1]
                altered_params[sys.argv[prev]] = [int(arg) if arg.isnumeric() else arg for arg in sys.argv[prev+1:idx]]

        train(model_name, epochs = epochs, hypothesis = hypothesis, altered_params = altered_params)
    else:
        train(model_name)
