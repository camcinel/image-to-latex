# Build and return the model here based on the configuration.
from baseline_model import *
from altered_LSTM import *
from RNN import *

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    model_type = config_data['model']['model_type']
    layer_num = config_data["model"]["layer_num"]
    
    if model_type == "baseline":
        embedding_size = config_data['model']['embedding_size']
        model = baseline(len(vocab), embedding_size, hidden_size, layer_num)
        
    elif model_type == 'A_LSTM':
        img_embedding_size = config_data['model']['img_embedding_size']
        word_embedding_size = config_data['model']['word_embedding_size']
        model = A_LSTM(len(vocab), img_embedding_size, word_embedding_size, hidden_size, layer_num)

    elif model_type == 'RNN':
        embedding_size = config_data['model']['embedding_size']
        model = RNN(len(vocab), embedding_size, hidden_size, layer_num)
        
    return model
