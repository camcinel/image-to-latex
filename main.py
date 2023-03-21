from experiment import Experiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    print('Beginning Training Loop')
    exp.run()
    print('Beginning Testing Loop')
    exp.test()
    print('Testing Temperatures')
    exp.temp_tests([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    print('Making attention image')
    exp.give_attention_image()
