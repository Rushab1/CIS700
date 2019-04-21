import csv
import pickle as pkl
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm



def parse_sentence(sentence, model):
    return predictor.predict(sentence)['verbs']



predictor = Predictor.from_path('../../../../Projects/models/allennlp/srl-model-2018.05.25.tar.gz')

with open('results.csv') as f:
    csvreader = csv.reader(f, delimiter='|')
    next(csvreader)
    parsed_descriptions = dict()
    for row in tqdm(csvreader):
        verbs = parse_sentence(row[2].strip(), predictor)
        if(row[0] in parsed_descriptions):
            parsed_descriptions[row[0]].append(verbs)
        else:
            parsed_descriptions[row[0]] = []

