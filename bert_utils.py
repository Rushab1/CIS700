from tqdm import tqdm
import pickle as pkl
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from IPython import embed

class ApnaBert:
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.to('cuda')
        for params in self.model.parameters():
            params.requires_grad = False



    def pad(self, sentences):
        maxlen = max([len(sentence) for sentence in sentences])
        sentences = [sentence + [0]*(maxlen - len(sentence)) for sentence in sentences]
        return sentences


    def get_bert_features(self, sentences):
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        indexed_tokens = self.pad([self.tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences])
        tokens_tensor = torch.tensor(indexed_tokens)
        tokens_tensor = tokens_tensor.to('cuda')
        embedding = self.model(tokens_tensor, output_all_encoded_layers=False)[1]
#        embedding.requires_grad = False
        return embedding
