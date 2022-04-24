wget -O data/Flickr8k_Dataset.zip "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
wget -O data/Flickr8k_text.zip "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
unzip -qq data/Flickr8k_Dataset.zip -d data/
unzip -qq data/Flickr8k_text.zip -d data/
rm data/Flickr8k_Dataset.zip
rm data/Flickr8k_text.zip