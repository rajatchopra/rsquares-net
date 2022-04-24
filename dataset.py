from collections import Counter

import pandas as pd
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence

from params import DATA_PATH


class TextField:

    def __init__(self, lang="en", lower=True):
        self.lower = lower
        self.tokenizer = get_tokenizer("spacy", language=lang)
    
    def tokenize(self, text):
        return [token.lower() if self.lower else token for token in self.tokenizer(text)]
    
    def build_vocab(self, sentences, min_freq=None):
        counter = Counter()
        for sentence in sentences:
            counter.update(self.tokenize(sentence))
        if min_freq is not None:
            counter = Counter({k: c for k, c in counter.items() if c >= min_freq})
        self.vocab = Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    
    def numericalize(self, text):
        return [
            self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi["<unk>"]
            for token in self.tokenize(text)
        ]


class FlickrDataset(Dataset):

    def __init__(self, data_type="train", transforms=None, lang="en", lower=True, min_freq=5, text_field=None):

        if data_type == "train":
            images_df  = pd.read_csv(DATA_PATH/"Flickr_8k.trainImages.txt", sep="\t", header=None, names=["image_filename"])
        elif data_type == "dev":
            images_df  = pd.read_csv(DATA_PATH/"Flickr_8k.devImages.txt", sep="\t", header=None, names=["image_filename"])
        elif data_type == "test":
            images_df  = pd.read_csv(DATA_PATH/"Flickr_8k.testImages.txt", sep="\t", header=None, names=["image_filename"])
        else:
            raise NotImplementedError("Data type must be one of train|dev|test")
        
        captions_df = pd.read_csv(DATA_PATH/"Flickr8k.token.txt", sep="\t", header=None, names=["image_filename#image_num", "caption"])
        captions_df = captions_df.join(
            captions_df["image_filename#image_num"].str.split("#", expand=True).rename(columns={0: "image_filename", 1: "image_num"})
        )
        captions_df = captions_df.merge(images_df, on="image_filename", how="inner").reset_index(drop=True)

        self.image_nums = captions_df["image_num"]
        self.image_filenames = captions_df["image_filename"]
        self.captions = captions_df["caption"]

        if text_field is None:
            self.text_field = TextField(lang=lang, lower=lower)
            self.text_field.build_vocab(self.captions.tolist(), min_freq=min_freq)
        else:
            self.text_field = text_field

        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_num = self.image_nums[idx]
        image_filename = self.image_filenames[idx]
        caption = self.captions[idx]

        image = Image.open(str(DATA_PATH/f"Flicker8k_Dataset/{image_filename}")).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)
        
        numericalized_caption = (
            [self.text_field.vocab.stoi["<sos>"]] +
            self.text_field.numericalize(caption) +
            [self.text_field.vocab.stoi["<eos>"]]
        )

        return image_filename, image_num, caption, image, torch.tensor(numericalized_caption)


class FlickrCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        image_filenames = [item[0] for item in batch]
        image_nums = [item[1] for item in batch]
        text_captions = [item[2] for item in batch]
        images = [item[3].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[4] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_idx)

        return image_filenames, image_nums, text_captions, images, captions


def get_data_loader(
    data_type,
    transforms,
    lang,
    lower,
    min_freq, 
    text_field,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(data_type, transforms, lang, lower, min_freq, text_field)
    pad_idx = dataset.text_field.vocab.stoi["<pad>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=FlickrCollate(pad_idx=pad_idx),
    )

    return loader, dataset
