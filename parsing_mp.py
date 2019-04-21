import csv
import pickle as pkl
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
import numpy as np
import multiprocessing as mp


predictor = Predictor.from_path('../../../../Projects/models/allennlp/biaffine-dependency-parser-ptb-2018.08.23.tar.gz')

def parse_sentence(sentence, model):
    return model.predict(sentence)

def parse_sentences(JobQueue, sent_list):
    parsed_descriptions = dict()
    for row in tqdm(sent_list):
        verbs = parse_sentence(row[2].strip(), predictor)
        if(row[0] in parsed_descriptions):
            parsed_descriptions[row[0]].append(verbs)
        else:
            parsed_descriptions[row[0]] = [verbs]
            
    JobQueue.put(parsed_descriptions)

def combine_dicts(JobQueue):
    all_parsed_descriptions = {}
    parsed_descriptions = JobQueue.get()
    
    while parsed_descriptions != "kill":
        #TODO: append all (key, value ) pairs in parsed_descriptions to all_parsed descriptions
        for key in parsed_descriptions.keys():
            all_parsed_descriptions[key] = parsed_descriptions[key]
        parsed_descriptions = JobQueue.get()
    with open('parsed_descriptions.pkl', 'wb') as f:
        pkl.dump(all_parsed_descriptions, f)

        
def parallel_parse_image_descriptions():
    descriptions = open('results.csv').read().split("\n")
    n = len(descriptions)
    h = 1000
    
    for i in range(0, n):
        descriptions[i] = descriptions[i].split("| ")
        
    manager = mp.Manager()
    JobQueue = manager.Queue()
    pool = mp.Pool()
    jobs = []
    
    writer = pool.apply_async(combine_dicts, (JobQueue, ))
    
    for i in range(0, int(np.ceil(n/h))):
        s = i * h
        e = min((i+1)*h, n)
        job = pool.apply_async(parse_sentences, (JobQueue, descriptions[s:e]))
        jobs.append(job)
        
    for job in jobs:
        job.get()
    JobQueue.put("kill")
    writer.get()
    
    pool.close()
    pool.join()


mera_dict = parallel_parse_image_descriptions()
