# Image To Latex

To run this experiment, use 

## Sample Configs
Three models are implemented in this projects:
* CALSTM network, the sample config is `calstm.json`
* ResNet-Transformer network, the sample config is `resnet_transformer.json`, it's the default config.
* ViT network, the sample config is `vit.json`
* ResNet-GlobalContext, the sample config is `resnet_gc.json`

All config changes can be made in configs/ViT.json

Run the follwing code block to download the dataset

    !wget https://storage.googleapis.com/i2l/data/dataset5.tgz
    !tar zxvf dataset5.tgz
    !rm -f dataset5.tgz

