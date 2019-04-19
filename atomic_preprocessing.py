#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.tokenize import word_tokenize, sent_tokenize
from visual_genome import api as vg
import multiprocessing as mp
# from utils import *
import pandas as pd
import numpy as np
import pickle
import requests
import csv
import spacy
import json
import ijson
import sys
import os


# In[2]:


def load_atomic_df():
    df = pd.read_csv('v4_atomic_all_agg.csv')
    return df

def create_df_dict_by_events(df):
    df_dct = {}
    for index, row in df.iterrows():
        df_dct[row["event"]] = row
    return df_dct


# Observations:
# - Events:
#       - All events start with "PersonX
#       - All events
# 

# In[3]:


#relevant small functions
def get_number_of_entities(event):
    if "PersonX" in event and "PersonY" in event and "PersonZ" in event:
        return 3
    if "PersonX" in event and "PersonY" in event:
        return 2
    return 1

def get(verb):
    output = []
    cnt = 0
    for index, row in df.iterrows():
        if verb in row["event"]:
            output.append(row)
    return output
# get("umbrella")


# In[4]:


#Extract relations from json for VISUAL GENOME
def get_relation_from_relation_dict(relation):
    relation_str = ""
    
    #########Extract Subject#########
    if "name" in relation['subject']:
        relation_str += relation['subject']['name']
        
    elif "names" in relation['subject']:
        if len(relation['subject']['names']) == 1:
            relation_str += str(relation['subject']['names'][0])
        else:
            relation_str += " ".join(relation['subject']['names'])
    else:
        relation_str += str(relation['subject'])
    
    #########Extract Predicate#########
    relation_str += " " + str(relation["predicate"]) + " "
    
    #########Extract Object#########
    if "name" in relation['object']:
        relation_str +=  relation['object']['name']
    elif "names" in relation['object']:
        if len(relation['object']['names']) == 1:
            relation_str += str(relation['object']['names'][0])
        else:
            relation_str +=  " ".join(relation['object']['names'])
    else:
        relation_str += str(relation['oject'])
    
    #########process#########
    relation_str = relation_str.lower()
    return relation_str
    
def create_relations_dict(df, images_per_json_file = 1000, clear_previous_dicts = True):
    file = open("./Data/relationships.json")
    relations = ijson.items(file, "item")

    if not os.path.exists("modelfiles"):
        os.mkdir("modelfiles")
        
    if not os.path.exists("./modelfiles/relations_dct"):
        os.mkdir("./modelfiles/relations_dct")
    
    if not os.path.exists("./modelfiles/data_maps/"):
        os.mkdir("./modelfiles/data_maps")
    
    if clear_previous_dicts == True:
        os.system("rm -rf modelfiles/relations_dct/*")
    
    relations_dct = {}
    image_id_vs_file_map = {}
    cnt = 0
    
    for relation in relations:
        cnt += 1
        if cnt % images_per_json_file == 0:
            sys.stdout.write("Completed extracting relations from " + str(cnt) + " images\r")
            sys.stdout.flush()
            
            json.dump(relations_dct, open("./modelfiles/relations_dct/" + str(cnt/images_per_json_file) + ".json" , "w"))
            
            for image_id in relations_dct.keys():
                image_id_vs_file_map[image_id] = "./modelfiles/relations_dct/" + str(cnt/images_per_json_file) + ".json"
                
            del relations_dct
            relations_dct = {}
    
        relation_list = []
        for r in relation["relationships"]:
            relation_list.append(get_relation_from_relation_dict(r))
        
        relations_dct[relation["image_id"]] = {}
        relations_dct[relation["image_id"]]["relations"] = relation_list
        vg.get_QA_of_image(id=61512)
        a=vg.get_QA_of_image(id=int(relation["image_id"]))
#         relations_dct[relation["image_id"]]["qa"] = 
        
    json.dump(image_id_vs_file_map, open("./modelfiles/data_maps/relations_id_vs_file_map.json", "w"))

a = create_relations_dict(df, 1000, clear_previous_dicts=True)
print(a)


# In[ ]:





# In[5]:


###########Overlap between relations from VG and events from ATOMIC###########
THRESHOLD = 0.8

def get_image_ids_by_overlap_relations_vs_events_singleFile(df_dct, relations_json_file, saveFile):
    file = open(relations_json_file)
    relations = ijson.items(file, "item")
    image_relation_map = {}
    
    cnt = 1
    for image_id in relations:
        relation_words = set(" ".join(relations[image_id]).split())
        cnt += 1
        
        if cnt % 10 == 0:
            print(relations_json_file + ": Done with " + cnt + " images\r")
            sys.stdout.write(relations_json_file + ": Done with " + cnt + " images\r")
            sys.stdout.flush()
            break
            
        for event in df_dct.keys():
            event_words = event.replace("PersonX", "")
            event_words = event_words.replace("PersonY", "")
            event_words = event_words.replace("PersonZ", "")
            
            event_words = set(["man", "woman", "person", "people", "girl", "boy", "child", "baby" ].extend(event_words.split()))

            intersection = event_words.intersection(relation_words)
            
            if len(intersection) >= THRESHOLD * len(event_words):                
                image = vg.get_image_data(id=image_id)
                
                image_relation_map[event] = {
                                                "event_effect": df_dct[event],
                                                "image_id": image_id,
                                                "image_url": image.url,
                                                "relations": relations[image_id],                        
                                            }
    json.dump(image_relation_map, open(saveFile, "w"))
    
def get_image_ids_by_overlap_relations_vs_events(df, dir = "./modelfiles/relations_dct/", saveDir = "./modelfiles/images_by_relation/"):
    df_dct = create_df_dict_by_events(df)
    
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
        
    if not saveDir.endswith("/"):
        saveDir += "/"
    
    if not dir.endswith("/"):
        dir += "/"
    
    pool = mp.Pool()
    jobs = []
    
    for file in os.listdir(dir):
        job = pool.apply_async(get_image_ids_by_overlap_relations_vs_events_singleFile, (df_dct, dir + file, saveDir + file,))
        jobs.append(job)
        
    for job in jobs:
        job.get()
    
    pool.close()
    pool.join()
        
# get_image_ids_by_overlap_relations_vs_events(df)


# In[ ]:


exec(open("utils.py").read())
from tqdm.auto import tqdm
###########FLICKR30k Images###########
def get_image_description_dct():
    img_decriptions = np.genfromtxt("./Data/flickr30k_images/results.csv", delimiter="| ", dtype="|S80", skip_header=1)
    descriptions_dct = {}
    
    for line in img_decriptions:
        image_id = line[0]
        
        try:
            descriptions_dct[image_id]
        except:
            descriptions_dct[image_id] = []
            
        descriptions_dct[image_id].append(line[2])
        
    return descriptions_dct


def get_overlap():
    df = load_atomic_df()
    df_dct = create_df_dict_by_events(df)
    image_descriptions = get_image_description_dct()
    overlap_dct = {}
    
    event_pos = {}
    image_descriptions_pos = {}
    
    for event in tqdm(df_dct.keys()):
        event_pos[event] = get_verbs([event])[0]
    
    for image_id in tqdm(image_descriptions):
        image_descriptions_pos[image_id] = get_verbs(image_descriptions[image_id])
    
    for event in tqdm(df_dct.keys()):
        for image_id in image_descriptions:
            overlap = compare(image_descriptions_pos[image_id], event_pos[event])
            overlap_dct[(event, image_id)] = overlap
    
    pickle.dump(overlap_dct, open("over_image_event.pkl", "w"))
    
get_overlap()


# In[16]:


from visual_genome import api
image = vg.get_image_data(id=23)
print ( image)

