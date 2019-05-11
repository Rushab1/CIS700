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

def extractor_reader(directory, image_files, extractor, JobQueue):
    ex = YouTube8MFeatureExtractor(model_dir="./modelfiles")
    for image_file in tqdm(image_files):
        im = Image.open(os.path.join( directory, image_file))
        im = np.array(im)
        features = ex.extract_rgb_frame_features(im)
        extractor.features_dict[image_file] = features
        # JobQueue.put((image_file, features))

def writer(extractor, JobQueue):
    while (1):
        res = JobQueue.get()
        if res == "kill":
            break
        image_file = res[0]
        features = res[1]
        extractor.features_dict[image_file] = features

def parallel_extraction_of_flickr_featres(image_directory, savefile):
    extractor = Extractor()
    manager = mp.Manager()
    pool = mp.Pool()
    JobQueue = manager.Queue()
    
    image_files = os.listdir(image_directory)

    n = len(image_files)
    h = 10

    jobs = []
    # writer_job = pool.apply_async(writer, (extractor, JobQueue))

    # for i in range(0, int(n/h)):
        # s = i * h
        # e = min((i+1)*h, n)

        # files_subset = image_files[s:e]
        # job = pool.apply_async(extractor_reader, (image_directory, files_subset, extractor, JobQueue))
        # jobs.append(job)

        # extractor_reader(image_directory, files_subset, extractor, [])

    # print(jobs)
    # for job in jobs:
        # job.get()

    # JobQueue.put("kill")
    # writer.get()
    # pool.close()
    # pool.join()

    extractor_reader(image_directory, image_files, extractor, [])
    pickle.dump(extractor.features_dict, open(savefile, "wb"))

if __name__ == "__main__":
    parallel_extraction_of_flickr_featres("../../Data/flickr-image-dataset/flickr30k_images/flickr30k_images/", "../../Data/features.pkl")
