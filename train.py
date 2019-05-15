import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
import argparse
import numpy as np
import random
from model import *
from sklearn.metrics import accuracy_score
from batch import *
from test import *

torch.manual_seed(1000)
random.seed(1000)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='pytorch_image_captioning/modelfiles/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='pytorch_image_captioning/modelfiles/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='pytorch_image_captioning/modelfiles/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--dataset', type=str, default = "all_train")
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument("-num_epochs", type = int, default = 100)
    parser.add_argument("-dropout", type = float, default = 0.5)
    parser.add_argument("-initial_lr", type = float, default = 0.05)
    opts = parser.parse_args()

    print("Parsing args: Done")

    print("Initializing Models")
    batch = Batch(opts, dataset = opts.dataset)
    intent_model = IntentModel()
    intent_model = intent_model.cuda()
    optim = Optim(intent_model, lr = 0.0001, weight_decay = 0) 
    print("Done")
    print("Starting Training")

    if not os.path.exists("./modelfiles/checkpoints"):
        os.mkdir("./modelfiles/checkpoints/")
        

    epoch_num = 0
    while epoch_num < opts.num_epochs:
            input, options, target, epoch_end_flag = batch.next_batch()

            input = input.cuda()
            options = options.cuda()
            target = target.cuda()

            pred = intent_model(input, options, target)
            optim.backward(pred, target)
            
            if epoch_end_flag:
                epoch_num += 1
                print("Epoch " + str(epoch_num) + " done: loss=" + str(optim.epoch_loss))
                optim.epoch_loss = 0

            if epoch_num % 5 == 0 and epoch_end_flag:
                pickle.dump(intent_model, open("modelfiles/checkpoints/checkpoint_" + str(epoch_num) + ".pkl" , "wb"))

#                if epoch_num > 100:
#                    optim.update_lr(optim.lr/1.05)
#                else:
#                    optim.update_lr(optim.lr/1.2)
'''
            if epoch_num % 10 == 0 and epoch_end_flag:
                print("____________________________________________")
                for property in properties:
                    print("Epoch " + str(epoch_num) + ":\t" + property +":\t" +
                        str(round(
                            test(model, batch, zero_shot=True, zero_shot_property=property,batch_from="dev",no_reverse=opts.no_reverse),
                            2)) + ":" +   
                        str(round(
                            test(model, batch, zero_shot=True, zero_shot_property=property, batch_from="test", no_reverse=opts.no_reverse)
                            , 2))) 

                print("Epoch " + str(epoch_num) + ":overall:" + 
                          str(test(model, batch, batch_from="dev", no_reverse=opts.no_reverse )) + ":" + 
                          str(test(model, batch, batch_from="test", no_reverse=opts.no_reverse )))
                # print(testByPole(model, batch, batch_from="test" ))
                pickle.dump(model, open("model.pkl", "wb"))

            elif epoch_num % 20 == 0 and epoch_end_flag and opts.dataset == "PCE":
                print("Epoch " + str(epoch_num) + ":overall:" + 
                          str(test(model, batch, batch_from="test", no_reverse=opts.no_reverse )))
                print(testByPole(model, batch, batch_from="test" ))
                pickle.dump(model, open("model.pkl", "wb"))
'''
