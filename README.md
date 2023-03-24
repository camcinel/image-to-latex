# Image To Latex
Image tp Latex final project for CSE-251B in UCSD, four models implemented. Best run ResNet-Transformer can get 93% BLEU-1 score.

## Usage
* Please run the cell in `get_dataset.ipynb` to download the dataset.
* Define the configuration for your experiment. See `.json` files to see the structure and available options.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Sample Configs
Three models are implemented in this projects:
* CALSTM network, the sample config is `calstm.json`
* ResNet-Transformer network, the sample config is `resnet_transformer.json`, it's the default config.
* ViT network, the sample config is `ViT.json`
* ResNet-GlobalContext, the sample config is `resnet_gc.json`

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- make_dataset: A simple implementation of `torch.utils.data.Dataset` the Image2Latex 140K
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace

## Acknowledges
* The file structure is adapted from PA4 of the class.
* The Resnet-Transformer model is adapted from https://github.com/kingyiusuen/image-to-latex, we changed position encodin methods and running on a different dataset.