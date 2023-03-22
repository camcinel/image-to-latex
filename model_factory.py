# Build and return the model here based on the configuration.
from vit import MathEquationConverter as vit_mec

class ModelSummary():
    def __init__(self, model):
        self.__enc_params = self.__count_parameters(model.encoder)
        self.__dec_params = self.__count_parameters(model.decoder)
        self.__total_params = self.__enc_params + self.__dec_params
    
    def __count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def __call__(self):
        print("\n------------------Model Summary----------------------")
        print(f'Trainable encoder parameters:\t{self.__enc_params:,}')
        print(f'Trainable decoder parameters:\t{self.__dec_params:,}')
        print(f'Total trainable parameters:\t{self.__total_params:,}')
        print("-----------------------------------------------------\n")
        
        
def get_model(config_data, vocab):
    model_type = config_data['model_type']
    max_length = config_data["generation"]["max_length"]
    
    if model_type == "default_experiment":
        raise NotImplementedError(f"model type {model_type} not implemented!")
    
    elif model_type == "ResTransformer_experiment":
        raise NotImplementedError(f"model type {model_type} not implemented!")

    elif model_type == "ViT": 
        # pack config_data['decoder'], num_classes, max_len into a single dictionary
        model = vit_mec(config_encoder=config_data['encoder'], 
                        config_decoder={
                            **config_data['decoder'], 
                            'num_classes': len(vocab), 
                            'max_len': max_length
                        })        
    else:
        raise ValueError(f" Unexpected model type {model_type}!")
    
    ModelSummary(model)() # print model summary

    return model
