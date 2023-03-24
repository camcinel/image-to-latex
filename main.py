from experiment import Experiment
import sys
import torch, gc

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = None
    model_types = ['calstm', 'resnet_gc', 'resnet_transformer', 'vit']

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    
    if exp_name not in model_types:
        raise ValueError(f"Need a model type in {model_types}, got {exp_name}!")

    print("Running Experiment: ", exp_name)

    exp = Experiment(exp_name)
    print('Beginning Training Loop')
    exp.run()
    print('Beginning Testing Loop')
    exp.test()
    
    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()