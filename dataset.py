from collections import Counter
from pathlib import Path
import pandas as pd
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import nltk

DATA_PATH = Path("data")


class FlickrDataset(Dataset):

    def __init__(self, data_type, vocab, transforms=None):

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
        self.image_nums = captions_df["image_num"].tolist()
        self.image_filenames = captions_df["image_filename"].tolist()
        self.captions = captions_df["caption"].tolist()
        
        self.vocab = vocab
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
            
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        numericalized_caption = []
        numericalized_caption.append(self.vocab("<sos>"))
        numericalized_caption.extend([self.vocab(token) for token in tokens])
        numericalized_caption.append(self.vocab("<eos>"))

        return image_filename, image_num, caption, image, torch.Tensor(numericalized_caption).to(torch.long)


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
    vocab,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(data_type, vocab, transforms)
    pad_idx = vocab("<pad>")
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=FlickrCollate(pad_idx=pad_idx),
    )

    return loader