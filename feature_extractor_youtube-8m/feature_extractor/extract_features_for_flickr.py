import multiprocessing as mp
from PIL import Image
from feature_extractor import *
from tqdm import tqdm
import numpy as np
import pickle
import os
import sys

class Extractor:
    if not os.path.exists("./modelfiles"):
        os.mkdir("modelfiles")
    # extractor = YouTube8MFeatureExtractor(model_dir="./modelfiles")
    features_dict = {}

def extractor_reader( image_files, extractor):
    ex = YouTube8MFeatureExtractor(model_dir="./modelfiles")
    for image_file in tqdm(image_files):
        im = Image.open( image_file)
        im = np.array(im)
        features = ex.extract_rgb_frame_features(im)
        extractor.features_dict[image_file] = features

if __name__ == "__main__":
    extractor = Extractor()
    
    image_dir = "/scratch/images_needed_coco/train/"
    image_list = os.listdir(image_dir)
    image_list = [image_dir + i for i in image_list]

    image_dir = "/scratch/images_needed_coco/valid/"
    image_local = os.listdir(image_dir)
    image_local = [image_dir + i for i in image_local]
    image_list.extend(image_local)

    image_dir = "/scratch/images_needed_flickr/"
    image_local = os.listdir(image_dir)
    image_local = [image_dir + i for i in image_local]
    image_list.extend(image_local)

    print(len(image_list))

    extractor_reader(image_list, extractor)
    pickle.dump(extractor.features_dict, open("/scratch/CIS700/Data/features_youtube.pkl", "wb"))
