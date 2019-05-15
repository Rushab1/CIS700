import os
import torch
import pickle
import argparse
import numpy as np
import random
import sys
sys.path.insert(0, "pytorch_image_captioning")
from sample import *
from bert_utils import *

#COCO_TRAIN_DIR = "/scratch/images_needed_coco/train/"
#COCO_VALID_DIR = "/scratch/images_needed_coco/valid/"
#FLICKR_TRAIN_DIR = "/scratch/images_needed_coco/flickr/train"
#FLICKR_VALID_DIR = "/scratch/images_needed_coco/flickr/valid"

#COCO_INTENT_TRAIN_DICT = "/scratch/data/coco/train/coco_intent_train.pkl"
#COCO_INTENT_VALID_DICT = "/scratch/data/coco/valid/coco_intent_valid.pkl"
#FLICKR_INTENT_TRAIN_DICT = "/scratch/data/flickr/train/flickr_intent_train.pkl"
#FLICKR_INTENT_VALID_DICT = "/scratch/data/flickr/valid/flickr_intent_valid.pkl"

#Downsampled Files
COCO_INTENT_TRAIN_DICT = "/scratch/data/coco/train/downsampled_coco_intent_train.pkl"
COCO_INTENT_VALID_DICT = "/scratch/data/coco/valid/downsampled_coco_intent_valid.pkl"
FLICKR_INTENT_TRAIN_DICT = "/scratch/data/flickr/train/downsampled_flickr_intent_train.pkl"
FLICKR_INTENT_VALID_DICT = "/scratch/data/flickr/valid/downsampled_flickr_intent_valid.pkl"

#Downsampled Files
COCO_INTENT_TRAIN_DICT = "/scratch/data/coco/train/downsampled_20_coco_intent_train.pkl"
COCO_INTENT_VALID_DICT = "/scratch/data/coco/valid/downsampled_20_coco_intent_valid.pkl"
FLICKR_INTENT_TRAIN_DICT = "/scratch/data/flickr/train/downsampled_20_flickr_intent_train.pkl"
FLICKR_INTENT_VALID_DICT = "/scratch/data/flickr/valid/downsampled_20_flickr_intent_valid.pkl"

FEATURES_CAPTIONING_FILE = "/scratch/CIS700/Data/features_captioning/features_captioning_all.pkl"
FEATURES_YOUTUBE_FILE = "/scratch/CIS700/Data/features_youtube.pkl"

BERT_EMBEDDING_FILE = "/scratch/pkl_files/bert_embeddings.pkl"

class Batch():
    images_list = []
    batch_size = 0
    local_cnt = 0
    dataset_size = 0
#    captioning_model = Model()
    print("Loading features captioning dict")
    features_captioning_dct = pickle.load(open(FEATURES_CAPTIONING_FILE, "rb"))
    print("Done")

    print("Loading features youtube dict")
    features_youtube_dct = pickle.load(open(FEATURES_YOUTUBE_FILE, "rb"))
    print("Done")

#    bert_model = ApnaBert()
    bert_dct  = pickle.load(open(BERT_EMBEDDING_FILE, "rb"))

    def __init__(self, args, dataset = "all_train", batch_size = 64):
        self.batch_size = batch_size
        if dataset == "coco_train" or dataset == "all_train":
            images_dct = pickle.load(open(COCO_INTENT_TRAIN_DICT, "rb"))
            self.images_list.extend(images_dct["questions"])

        if dataset == "coco_valid" or dataset == "all_valid":
            images_dct = pickle.load(open(COCO_INTENT_VALID_DICT, "rb"))
            self.images_list.extend(images_dct["questions"])
            
        if dataset == "flickr_train" or dataset == "all_train":
            images_dct = pickle.load(open(FLICKR_INTENT_TRAIN_DICT, "rb"))
            self.images_list.extend(images_dct["questions"])

        if dataset == "flickr_valid" or dataset == "all_valid":
            images_dct = pickle.load(open(FLICKR_INTENT_VALID_DICT, "rb"))
            self.images_list.extend(images_dct["questions"])
        
        try:
            self.dim = args.dim
        except:
            self.dim = 256

        try:
            self.bert_dim = args.bert_dim
        except:
            self.bert_dim = 768

        self.dataset_size = len(self.images_list)
        print("Dataset Size: " + str(self.dataset_size))
#        self.captioning_model.load_model(args)
    
    def random_shuffle(self):
        random.shuffle(self.images_list)

    def epochEnd(self):
        self.local_cnt = 0
        self.random_shuffle() 

    def extract_youtube_features(self, image_paths_list):
        features = []
        for img in image_paths_list:
            try:
                tmp = self.features_youtube_dct[img]
                #embed()
                features.append(torch.FloatTensor(tmp).cuda())
            except:
                features.append(torch.rand(1024).cuda())

        features = torch.stack(features, 0)

        return features

        pass

    def extract_captioning_features(self, image_paths_list):
        features = []
        captions = []
        for img in image_paths_list:
            try:
                features.append(self.features_captioning_dct[img]['features'])
            except:
                features.append(torch.rand(self.dim).cuda())

            try:
                captions.append(self.features_captioning_dct[img]['captions'])
            except:
                captions.append("<no_key>")


        features = torch.stack(features, 0)

#        features, captions, eos = self.captioning_model.generate_caption_list("", image_paths_list)
        
        caption_embedding = self.extract_text_bert_embedding(captions)
        
#        features = torch.rand(len(image_paths_list), 256)
#        caption_embedding = torch.rand(len(image_paths_list), 768)
        return features, caption_embedding

    def extract_text_bert_embedding(self, text_list):
        emb = []
        for i in text_list:
            try:
                emb.append(self.bert_dct[i].squeeze(0))
            except:
                emb.append(torch.rand(self.bert_dim).cuda())

        return torch.stack(emb, 0)
#        return self.bert_model.get_bert_features(text_list)

    def divide_chunks(l, n):
        for i in range(0, len(l), n):  
            yield l[i:i + n]


    def next_batch(self):
        try:
            s = self.local_cnt
        except:
            self.local_cnt = 0
            s = 0

        e = min(s + self.batch_size, self.dataset_size)

        local_image_paths = []
        targets = []
        options = []

        for i in range(s,e):
            local_image_paths.append(self.images_list[i]['image_path'])
            targets.append(self.images_list[i]['answer'] - 1)
            options.append(self.extract_text_bert_embedding(self.images_list[i]['options']))

        caption_features, caption_embeddings = self.extract_captioning_features(local_image_paths)

        youtube_features = self.extract_youtube_features(local_image_paths)

        #Changes here
        x1 = caption_features
        x2 = caption_embeddings
        x3 = youtube_features

#        x = torch.cat((x1,x2), 1)
#        x = x1
#        x = torch.rand(e-s, 256)
#        x = x2
        x = x3
        o = options
        t = torch.LongTensor(targets)
        
        self.local_cnt = e

        epoch_end_flag = False
        if e >= self.dataset_size:
            self.epochEnd()
            epoch_end_flag = True
        o = torch.stack(o, 0)

#        x1.requires_grad = False
#        o.requires_grad = False
#        t.requires_grad = False
        return x, o, t, epoch_end_flag
