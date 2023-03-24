# Build and return the model here based on the configuration.
from .modules.resnet_transformer import MathEquationConverter as RT_MEC
from .modules.vit import MathEquationConverter as VIT_MEC
from .modules.calstm import ImageToLatex
from .modules.resnet_gc import MathEquationConverter as RGC_MEC
from .utils.summary_utils import ModelSummary as summary

def get_model(config_data, vocab):
    model_type = config_data['model']['model_type']
    
    if model_type == "resnet_transformer":
        nhead = config_data['model']['nhead']
        layer_num = config_data["model"]["layer_num"]
        dropout = config_data["model"]["dropout"]
        dim_feedforward = config_data["model"]["dim_feedforward"]
        max_length = config_data["generation"]["max_length"]
        embedding_size = config_data['model']['embedding_size']
        model = RT_MEC(embedding_size, nhead, layer_num, dim_feedforward, dropout, len(vocab), max_length)

    elif model_type == "vit":
        # pack config_data['decoder'], num_classes, max_len into a single dictionary
        max_length = config_data["generation"]["max_length"]
        model = VIT_MEC(config_encoder=config_data['encoder'], 
                        config_decoder={
                            **config_data['decoder'],
                            'num_classes': len(vocab),
                            'max_len': max_length
                        })
        
    elif model_type == "resnet_gc":
        nhead = config_data['model']['nhead']
        layer_num = config_data["model"]["layer_num"]
        dropout = config_data["model"]["dropout"]
        dim_feedforward = config_data["model"]["dim_feedforward"]
        max_length = config_data["generation"]["max_length"]
        embedding_size = config_data['model']['embedding_size']
        model = RGC_MEC(embedding_size, nhead, layer_num, dim_feedforward, dropout, len(vocab), max_length)

    elif model_type == "calstm":
        hidden_size = config_data['model']['hidden_size']
        layer_num = config_data["model"]["layer_num"]
        embedding_size = config_data["model"]["embedding_size"]
        temperature = config_data["generation"]['temperature']
        determinism = config_data["generation"]['deterministic']
        max_length = config_data["generation"]['max_length']
        model = ImageToLatex(L=4 * 34, D=512, hidden_size=hidden_size, embedding_size=embedding_size, n_layers=layer_num,
                         vocab=vocab, temperature=temperature, determinism=determinism, max_length=max_length)

    else:
        raise ValueError(f" Unexpected model type {model_type}!")
        
    summary(model)() # print model summary
        
    return model