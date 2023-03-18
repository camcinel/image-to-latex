# Build and return the model here based on the configuration.
from vit import *

def get_model(config_data, vocab):
    # nhead = config_data['model']['nhead']
    model_type = config_data['model_type']
    # layer_num = config_data["model"]["layer_num"]
    # dropout = config_data["model"]["dropout"]
    # dim_feedforward = config_data["model"]["dim_feedforward"]
    max_length = config_data["generation"]["max_length"]
    
    # if model_type == "baseline":
    #     embedding_size = config_data['model']['embedding_size']
    #     model = MathEquationConverter(embedding_size, nhead, layer_num, dim_feedforward, dropout, len(vocab), max_length)
    
    if model_type == "ViT":
        # print(config_data["encoder"])
        model = MathEquationConverter(config_data['encoder'], config_data['decoder'], len(vocab), max_length)        
    
    return model
