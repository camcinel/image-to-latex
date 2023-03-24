# Build and return the model here based on the configuration.
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
