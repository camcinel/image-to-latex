# Build and return the model here based on the configuration.
from baseline_model import *

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    model_type = config_data['model']['model_type']
    layer_num = config_data["model"]["layer_num"]
    
    if model_type == "baseline":
        embedding_size = config_data['model']['embedding_size']
        model = MathEquationConverter(embedding_size, 4, 8, 338, 152)
        
    return model
