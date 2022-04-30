from collections import Counter
import pickle
import pandas as pd
import nltk


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    

def build_vocab(sentences, min_freq=None):
    counter = Counter()
    for sentence in sentences:
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        counter.update(tokens)
    if min_freq is not None:
        words = [k for k, c in counter.items() if c >= min_freq]
    
    vocab = Vocabulary()
    vocab.add_word('<pad>') # 0
    vocab.add_word('<sos>') # 1
    vocab.add_word('<eos>') # 2
    vocab.add_word('<unk>') # 3

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == "__main__":
    images_df  = pd.read_csv("data/Flickr_8k.trainImages.txt", sep="\t", header=None, names=["image_filename"])
    captions_df = pd.read_csv("data/Flickr8k.token.txt", sep="\t", header=None, names=["image_filename#image_num", "caption"])
    captions_df = captions_df.join(
        captions_df["image_filename#image_num"].str.split("#", expand=True).rename(columns={0: "image_filename", 1: "image_num"})
    )
    captions_df = captions_df.merge(images_df, on="image_filename", how="inner").reset_index(drop=True)
    
    vocab = build_vocab(captions_df["caption"].tolist(), min_freq=5)
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
