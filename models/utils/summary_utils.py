class ModelSummary():
    """
    Print the encoder/decoder and total parameters when calling any model. 
    """
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