import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from vocab import load_vocab
from make_dataset import *


# Builds your datasets here based on the configuration.
def get_datasets(config_data):
    imgs_root_dir = config_data['dataset']['images_root_dir']
    root_train = config_data['dataset']['train_data_dir']
    root_val = config_data['dataset']['val_data_dir']
    root_test = config_data['dataset']['test_data_dir']
    vocab_path = config_data['dataset']['vocab_dir']

    vocabulary = load_vocab(vocab_path)
    train_data_loader_lst = get_coco_dataloader(imgs_root_dir, root_train, config_data)
    val_data_loader_lst = get_coco_dataloader(imgs_root_dir, root_val, config_data)
    test_data_loader_lst = get_coco_dataloader(imgs_root_dir, root_test, config_data)

    return vocabulary, train_data_loader_lst, val_data_loader_lst, test_data_loader_lst


def get_coco_dataloader(imgs_root_dir, meta_data_path, config_data):
    loaders = []
    meta_data = pd.read_pickle(meta_data_path)
    padded_lengths = meta_data['padded_seq_len'].unique()
    np.random.seed(140)
    torch.manual_seed(140)
    np.random.shuffle(padded_lengths)
#     padded_lengths = padded_lengths[0:2]
    for padded_length in padded_lengths:
        data = meta_data[meta_data['padded_seq_len'] == padded_length]
        dataset = MyDataset(root=imgs_root_dir,
                            meta_data=data,
                            img_size=(config_data['dataset']['img_h'], config_data['dataset']['img_w'])
                            )
    
        loaders.append( DataLoader(dataset=dataset,
                        batch_size=config_data['dataset']['batch_size'],
                        shuffle=True,
                        num_workers=config_data['dataset']['num_workers'],
                        pin_memory=True) )
    return loaders