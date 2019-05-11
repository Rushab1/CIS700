import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from tqdm import tqdm


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model:
    encoder = None
    decoder = None
    transform = None
    vocab = None

    def load_model(self, args):
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
        
        # Load vocabulary wrapper
        with open(args.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Build models
        self.encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        self.decoder = DecoderRNN(args.embed_size, args.hidden_size, len(self.vocab), args.num_layers)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        # Load the trained model parameters
        self.encoder.load_state_dict(torch.load(args.encoder_path))
        self.decoder.load_state_dict(torch.load(args.decoder_path))

    def load_image(self, image_path, transform=None):
        image = Image.open(image_path)
        image = image.resize([224, 224], Image.LANCZOS)

        if transform is not None:
            image = transform(image).unsqueeze(0)
        
        return image

    def generate_caption(self, image_file):
        # Prepare an image
        image = self.load_image(image_file, self.transform)
        image_tensor = image.to(device)
        
        # Generate an caption from the image
        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        # print(image_tensor.shape, sampled_ids.shape)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        
        # Print out the image and the generated caption
        # print (sentence)
        image = Image.open(args.image)
        plt.imshow(np.asarray(image))

    def generate_caption_list(self, prefix, image_files):
        # Prepare images
        images = []
        for image_file in image_files:
            image = self.load_image(prefix + image_file, self.transform)
            image_tensor = image.to(device)
            images.append(image.squeeze(0))

        image_tensor = torch.stack(images, 0)
        
        # Generate an caption from the image
        feature = self.encoder(image_tensor)
        sampled_ids_list = self.decoder.sample(feature)

        captions = []       
        lstm_outputs = sampled_ids_list
        eos_pos = []
        
        for sampled_ids in sampled_ids_list:
            sampled_ids = sampled_ids.cpu().numpy()
            # Convert word_ids to words
            sampled_caption = []
            
            cnt = 0
            for word_id in sampled_ids:
                word = self.vocab.idx2word[word_id]
                sampled_caption.append(word)

                cnt += 1
                if word == '<end>':
                    break
            eos_pos.append(cnt)
            
            sentence = ' '.join(sampled_caption)
            
            # Print out the image and the generated caption
            captions.append(sentence)
            # print (sentence)
            # image = Image.open(args.image)
            # plt.imshow(np.asarray(image))

        return feature, captions, lstm_outputs, eos_pos

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--image_dir', type=str, help='input image directory for generating caption')
    parser.add_argument('--save_dir', type=str, help='output directory for generating caption', default='../Data/features_captioning/')

    parser.add_argument('--encoder_path', type=str, default='./modelfiles/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='modelfiles/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='modelfiles/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    # main(args)

    model = Model()
    model.load_model(args)
    features_dict = {}

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    image_dir = args.image_dir
    image_list = os.listdir(image_dir)
    n = len(image_list)
    h = 30 #batch_Size

    
    features_dict = {}
    output_file_indexes = {}
    cnt = 0
    file_cnt = 0
    images_per_file = 10

    for i in tqdm(range(0, int(1.0* n /h))):
        s = i * h
        e = min( (i+1)*h, n )

        features, captions, lstm_outputs, eos_pos = model.generate_caption_list(image_dir, image_list[s:e])
        
        for j in range(s,e):
            fname = image_list[j]
            features_dict[fname] = {}
            features_dict[fname]['features'] = features[j-s]
            features_dict[fname]['lstm_outputs'] = lstm_outputs[j-s]
            features_dict[fname]['captions'] = captions[j-s]
            features_dict[fname]['eos_pos'] = eos_pos[j-s]

            output_file_indexes[fname] = args.save_dir + "features_captioning_" + str(file_cnt) + ".pkl"

        cnt += h
        if cnt % images_per_file == 0:
            pickle.dump(features_dict, open(args.save_dir + "features_captioning_" + str(file_cnt) + ".pkl", "wb"))
            pickle.dump(output_file_indexes, open("image_file_map.pkl", "wb"))
            file_cnt += 1


