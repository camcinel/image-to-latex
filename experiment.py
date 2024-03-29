import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
from datetime import datetime

from utils.caption_utils import *
from constants import ROOT_STATS_DIR
from utils.dataset_factory import get_datasets
from utils.file_utils import *
from models.model_factory import get_model
from torch.optim.lr_scheduler import CosineAnnealingLR


# Class to encapsulate a neural experiment adpapted from PA4.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./configs/', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: s", name)

        self.__name = config_data['experiment_name']
        self.__model_type = config_data['model']['model_type']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.device =  torch.device('cuda' if torch.has_cuda else 'cpu')
        self.config_data = config_data
        # Load Datasets
        self.__vocab, self.__train_loaders, self.__val_loaders, self.__test_loaders = get_datasets(
            config_data)
        
        # when use CALSTM, add a padding token
        if self.__model_type == 'calstm':
            self.__beam_width = config_data['experiment']['beam_width']
            self.__vocab.idx2word[len(self.__vocab.idx2word)] = '\pad'
            self.__vocab.word2idx = {word: idx for idx, word in self.__vocab.idx2word.items()}

        # Setup Experiment
        self.__generation_config = config_data['generation']
        if 'beam_width' in config_data['generation'].keys():
            self.__beam_width = config_data['generation']['beam_width']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        if 'patience' in config_data['experiment']:
            self.__patience = config_data['experiment']['patience']
        else:
            self.__patience = 5
            
        # Init Model
        self.__model = get_model(config_data, self.__vocab).to(self.device)

        self.__best_model = copy.deepcopy(self.__model)  # Save your best model in this field and use this in test method.
    
        #Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        self.scheduler = CosineAnnealingLR(self.__optimizer, T_max=10, eta_min=1e-6)

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        
        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.json')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.json')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
            self.__best_model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)
            write_to_file_in_dir(self.__experiment_dir, 'config.json', self.config_data)
            


    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        cur_patience = 0
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            if self.__val_losses and val_loss > self.__val_losses[-1]:
                cur_patience += 1
            else:
                cur_patience = 0
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if cur_patience >= self.__patience:
                print(f'Early Stopped at: {epoch}!')
                break

    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = cnt = 0

        # Iterate over the data, implement the training function
        for data in self.__train_loaders:
            for i, (images, captions) in enumerate(data):
                self.__optimizer.zero_grad()
                images = images.contiguous().to(self.device)                             
                captions = captions.contiguous().to(self.device)
                output = self.__model(images, captions).contiguous().to(self.device)

                if self.__model_type != 'calstm':
                    output = output[:,:-1].contiguous().to(self.device)
                    captions = captions[:,1:].contiguous().to(self.device)                
                    loss = self.__criterion(output.view(-1, len(self.__vocab)), captions.view(-1))
                else:
                    loss = self.__criterion(output.transpose(1, 2), captions)

                loss.backward()
                self.__optimizer.step()
                training_loss += loss * images.size(0)
                cnt += images.size(0)
        self.scheduler.step()
        training_loss /= cnt
        return training_loss

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = cnt = 0

        with torch.no_grad():
            for data in self.__val_loaders:
                for i, (images, captions) in enumerate(data):
                    images = images.contiguous().to(self.device)
                    captions = captions.contiguous().to(self.device)
                    output = self.__model(images, captions).contiguous().to(self.device)
                    
                    output = output[:,:-1].contiguous().to(self.device)
                    captions = captions[:,1:].contiguous().to(self.device) 
                    
                    if self.__model_type != 'calstm':
                        output = output[:,:-1].contiguous().to(self.device)
                        captions = captions[:,1:].contiguous().to(self.device)                
                        loss = self.__criterion(output.view(-1, len(self.__vocab)), captions.view(-1))
                    else:
                        loss = self.__criterion(output.transpose(1, 2), captions)

                    val_loss += loss * images.size(0)
                    cnt += images.size(0)
        
        val_loss /= cnt
        if not self.__val_losses or val_loss < min(self.__val_losses):
            self.__save_best_model()
            self.__best_model = copy.deepcopy(self.__model)
        
        return val_loss

    #  Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    def test(self):
        self.__best_model.eval()
        test_loss = cnt = 0
        _bleu1 = 0
        _bleu4 = 0

        with torch.no_grad():
            for data in self.__test_loaders:
                for iter, (images, captions) in enumerate(data):
                    images = images.contiguous().to(self.device)
                    captions = captions.contiguous().to(self.device)
                    output_loss = self.__best_model(images, captions).contiguous().to(self.device)
                    if self.__model_type != 'calstm':
                        output_loss = output_loss[:,:-1].contiguous().to(self.device)
                        captions = captions[:,1:].contiguous().to(self.device)

                    test_loss += self.__criterion(output_loss.view(-1, len(self.__vocab)), captions.view(-1)) * images.size(0)
                    cnt += images.size(0)
                    
                    if 'beam_width' in config_data['generation'].keys():
                         if self.__beam_width > 1:
                            output = self.__best_model.predict_beamsearch(images, self.__beam_width)
                         else:
                            output = self.__best_model.predict(images)
                            
                    else:
                         output = self.__best_model.predict(images)
                            
                    for j in range(images.size(0)):
                        actual_cap = remove([self.__vocab.idx2word[x.item()] for x in captions[j]])
                        if self.__model_type != 'calstm':
                            output_cap = remove(get_caption(output[j], self.__vocab, self.__generation_config))
                        else:
                            output_cap = remove(output[j])
                        _bleu1 += bleu1([actual_cap], output_cap)
                        _bleu4 += bleu4([actual_cap], output_cap)

        _bleu1 /= cnt
        _bleu4 /= cnt
        test_loss /= cnt
                
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, _bleu1, _bleu4)
        self.__log(result_str)
        print(f'The actual cap is: {actual_cap} The prediction is:{output_cap}')
        print(f'teacher forcing preds: {remove(get_caption(torch.argmax(output_loss[-1],dim=-1), self.__vocab, self.__generation_config))}' )

        return test_loss, _bleu1, _bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __save_best_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss.item())
        self.__val_losses.append(val_loss.item())

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.json', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.json', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.close()
