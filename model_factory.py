# Build and return the model here based on the configuration.
from calstm import ImageToLatex


def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    model_type = config_data['model']['model_type']
    layer_num = config_data["model"]["layer_num"]
    embedding_size = config_data["model"]["embedding_size"]
    temperature = config_data["generation"]['temperature']
    determinism = config_data["generation"]['deterministic']
    max_length = config_data["generation"]['max_length']

    model = ImageToLatex(L=4 * 34, D=512, hidden_size=hidden_size, embedding_size=embedding_size, n_layers=layer_num,
                         vocab=vocab, temperature=temperature, determinism=determinism, max_length=max_length)

    return model
