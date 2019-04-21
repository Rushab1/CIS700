from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from IPython import embed

lemmatizer = WordNetLemmatizer()

def get_verbs(sentences):
    sentence_verb_pairs = []
    for sentence in tqdm(sentences):
        verbs = []
        nouns = []
        others = []
        pos_tags = nltk.pos_tag(word_tokenize(str(sentence).replace('PersonX', 'he')))
        for token, tag in pos_tags:
            if('VB' in tag and lemmatizer.lemmatize(token, pos='v') != 'be'):
                verbs.append(lemmatizer.lemmatize(token, pos='v'))
            elif('NN' in tag or 'PRP' in tag):
                nouns.append(token)
            else:
                others.append(token)
        sentence_verb_pairs.append({'verbs': verbs, 'nouns': nouns, 'others': others})

    return sentence_verb_pairs

def compare(descriptions, event):
    overlap = 0
    for description in descriptions:
        for verb in description['verb']:
            if(verb in event['verb']):
                overlap += 1
                break

    return overlap/len(descriptions)




# sentences = ["Jack kills Jill", "Ramu holds an umbrella"]

# embed()

