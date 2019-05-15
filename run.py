import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--image_dir', type=str, help='input image directory for generating caption')
    parser.add_argument('--save_dir', type=str, help='output directory for generating caption', default='../Data/features_captioning/')
    parser.add_argument('--dataset_type', type=str, help='output directory for generating caption', default='Coco')

    parser.add_argument('--encoder_path', type=str, default='./modelfiles/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='modelfiles/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='modelfiles/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    # main(args)
