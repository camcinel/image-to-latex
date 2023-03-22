# Build and return the model here based on the configuration.
from vit import *

def get_model(config_data, vocab):
    model_type = config_data['model_type']
    max_length = config_data["generation"]["max_length"]
    
    if model_type == "ViT":
        
        # pack config_data['decoder'], num_classes, max_len into a single dictionary
        
        model = MathEquationConverter(config_encoder=config_data['encoder'], 
                                      config_decoder={
                                          **config_data['decoder'], 
                                          'num_classes': len(vocab), 
                                          'max_len': max_length
                                      }
                                     )
        
    else:
        raise ValueError(f" model type expected ViT, got {model_type}!")
    
    return model
