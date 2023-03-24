# Build and return the model here based on the configuration.
from calstm import ImageToLatex
from models.ViT import MathEquationConverter as ViT_MEC
from utils.summary_utils import ModelSummary
        
        
def get_model(config_data, vocab):
    model_type = config_data['model_type']
    max_length = config_data["generation"]["max_length"]
    
    if model_type == "default_experiment":
        raise NotImplementedError(f"model type {model_type} not implemented!")
    
    elif model_type == "ResTransformer_experiment":
        raise NotImplementedError(f"model type {model_type} not implemented!")
        
    elif model_type == "ResNet_GC":
        raise NotImplementedError(f"model type {model_type} not implemented!")
    
    elif model_type == "calstm":
        hidden_size = config_data['model']['hidden_size']
        layer_num = config_data["model"]["layer_num"]
        embedding_size = config_data["model"]["embedding_size"]
        temperature = config_data["generation"]['temperature']
        determinism = config_data["generation"]['deterministic']

        model = ImageToLatex(L=4 * 34, D=512, hidden_size=hidden_size, embedding_size=embedding_size, n_layers=layer_num,
                             vocab=vocab, temperature=temperature, determinism=determinism, max_length=max_length)

    elif model_type == "ViT": 
        # pack config_data['decoder'], num_classes, max_len into a single dictionary
        model = ViT_MEC(config_encoder=config_data['encoder'], 
                        config_decoder={
                            **config_data['decoder'], 
                            'num_classes': len(vocab), 
                            'max_len': max_length
                        })        
    else:
        raise ValueError(f" Unexpected model type {model_type}!")
    
    ModelSummary(model)() # print model summary

    return model
