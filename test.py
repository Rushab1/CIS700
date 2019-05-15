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
    parser.add_argument('--dataset', type=str, default = "all_valid")
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--modelfile', type=str, default="./modelfiles/checkpoints/checkpoint_5.pkl") 
    opts = parser.parse_args()

    print("Parsing args: Done")
    test = Test(opts)
    test.test(opts.modelfile)

class Test:
    def __init__(self, opts):
        print("Initializing dependency Models")
        self.batch = Batch(opts, dataset = opts.dataset, batch_size = 1000000)
        print("Done")

        print("Loading Pretrained Model")
        self.intent_model = pickle.load(open(opts.modelfile, "rb"))
        self.intent_model = self.intent_model.cuda()
        self.intent_model.eval()
        print("Done")


    def test(self, modelfile):
        print("Predicting")
        epoch_num = 0
        global_target = []
        global_pred = []

        while True:
                input, options, target, epoch_end_flag = self.batch.next_batch()

                input = input.cuda()
                options = options.cuda()
                target = target.cuda()

                pred_prob = self.intent_model(input, options, target)
                _, pred = torch.max(pred_prob, 1)
                
                assert(len(pred) == len(target))
                global_target.extend(target.tolist())
                global_pred.extend(pred.tolist())
                
                if epoch_end_flag:
                    break

        print("Accuracy: " + str(accuracy_score(global_pred, global_target)))
        print("Total samples: " + str(len(global_target)))


