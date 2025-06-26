import numpy as np
import os
import pandas as pd
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import nltk
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK punkt for tokenization
nltk.download('punkt')

# Set paths (adjust as needed)
IMAGES_PATH = './data/Flickr8k_Dataset/Flicker8k_Dataset/'
CAPTIONS_PATH = './data/Flickr8k_text/Flickr8k.token.txt'
TRAIN_PATH = './data/Flickr8k_text/Flickr_8k.trainImages.txt'
TEST_PATH = './data/Flickr8k_text/Flickr_8k.devImages.txt'
GLOVE_PATH = './data/glove.6B.100d.txt'

# Load captions and create tokens dictionary
def load_captions():
    try:
        captions = open(CAPTIONS_PATH, 'r').read().split("\n")
        tokens = {}
        for line in captions:
            if not line:
                continue
            temp = line.split("#")
            if len(temp) < 2:
                continue
            img, cap = temp[0], temp[1][2:].lower()
            if img in tokens:
                tokens[img].append(cap)
            else:
                tokens[img] = [cap]
        return tokens
    except FileNotFoundError:
        raise FileNotFoundError(f"Captions file {CAPTIONS_PATH} not found")

tokens = load_captions()

# Load train and test image names
def load_image_lists():
    try:
        x_train = open(TRAIN_PATH, 'r').read().split("\n")
        x_test = open(TEST_PATH, 'r').read().split("\n")
        x_train = [x for x in x_train if x]
        x_test = [x for x in x_test if x]
        print(f"Number of Training Images: {len(x_train)}")
        print(f"Number of Test Images: {len(x_test)}")
        return x_train, x_test
    except FileNotFoundError:
        raise FileNotFoundError(f"Train or test file not found")

x_train, x_test = load_image_lists()

# Image preprocessing with augmentation
def preprocess_image(img_path, augment=False):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        if augment:
            aug = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )
            img = next(iter(aug.flow(img)))[0]
        img = preprocess_input_resnet(img)
        return img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Feature extraction using ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=resnet.input, outputs=resnet.output)

def get_image_encoding(img_name, augment=False):
    img_path = os.path.join(IMAGES_PATH, img_name)
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found")
        return None
    img = preprocess_image(img_path, augment=augment)
    if img is None:
        return None
    pred = feature_extractor.predict(img, verbose=0)
    return pred.squeeze()

# Encode images and save to datasets
def encode_and_save_images(image_list, output_file, dataset_file, augment=False):
    encoded_images = {}
    c_count = 0
    with open(dataset_file, 'w', encoding='utf-8') as f:
        f.write("image_id\tcaptions\n")
        for img in image_list:
            encoding = get_image_encoding(img, augment=augment)
            if encoding is None:
                continue
            encoded_images[img] = encoding
            for capt in tokens.get(img, []):
                caption = f"<start> {capt} <end>"
                f.write(f"{img}\t{caption}\n")
                c_count += 1
        f.flush()
    with open(output_file, 'wb') as f:
        pickle.dump(encoded_images, f)
    print(f"Saved {c_count} captions to {dataset_file}")

# Process train and test images
encode_and_save_images(x_train, "train_encoded_images.p", "flickr_8k_train_dataset.txt", augment=True)
encode_and_save_images(x_test, "test_encoded_images.p", "flickr_8k_val_dataset.txt", augment=False)

# Build vocabulary
def build_vocabulary(dataset_file):
    pd_dataset = pd.read_csv(dataset_file, delimiter='\t')
    ds = pd_dataset.values
    print(f"Dataset Shape: {ds.shape}")

    sentences = [row[1] for row in ds if isinstance(row[1], str)]
    print(f"Number of Sentences: {len(sentences)}")

    words = [word_tokenize(sent) for sent in sentences]
    word_counts = {}
    for sent in words:
        for word in sent:
            word_counts[word] = word_counts.get(word, 0) + 1

    min_freq = 5
    unique_words = [w for w in set(word for sent in words for word in sent) 
                    if word_counts[w] >= min_freq or w in ['<start>', '<end>', '<pad>']]
    print(f"Vocabulary Size (after filtering): {len(unique_words)}")

    word_2_indices = {w: idx + 1 for idx, w in enumerate(unique_words)}  # Reserve 0 for padding
    word_2_indices['<pad>'] = 0
    indices_2_word = {idx: w for w, idx in word_2_indices.items()}
    
    print(f"Index of <start>: {word_2_indices['<start>']}")
    print(f"Word at index {word_2_indices['<start>']}: {indices_2_word[word_2_indices['<start>']]}")

    return sentences, word_2_indices, indices_2_word, len(unique_words) + 1

sentences, word_2_indices, indices_2_word, vocab_size = build_vocabulary("flickr_8k_train_dataset.txt")

# Compute maximum caption length
max_len = max(len(word_tokenize(sent)) for sent in sentences)
print(f"Maximum Caption Length: {max_len}")

# Load GloVe embeddings
embedding_size = 100
embedding_matrix = np.zeros((vocab_size, embedding_size))
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in word_2_indices:
            embedding_matrix[word_2_indices[word]] = np.array(values[1:], dtype='float32')
np.save("embedding_matrix.npy", embedding_matrix)

# Create padded sequences and next words
def create_sequences(sentences, word_2_indices, max_len, vocab_size):
    padded_sequences = []
    subsequent_words = []
    for sent in sentences:
        text = word_tokenize(sent)
        text_ids = [word_2_indices.get(word, word_2_indices['<pad>']) for word in text]
        for i in range(1, len(text_ids)):
            padded_seq = text_ids[:i]
            next_word = text_ids[i]
            padded_seq = pad_sequences([padded_seq], maxlen=max_len, padding='post')[0]
            next_word_1hot = np.zeros(vocab_size, dtype=np.bool_)
            next_word_1hot[next_word] = 1
            padded_sequences.append(padded_seq)
            subsequent_words.append(next_word_1hot)
    
    return np.array(padded_sequences), np.array(subsequent_words)

padded_sequences, subsequent_words = create_sequences(sentences, word_2_indices, max_len, vocab_size)
print(f"Padded Sequences Shape: {padded_sequences.shape}")
print(f"Subsequent Words Shape: {subsequent_words.shape}")

# Load encoded images
with open('train_encoded_images.p', 'rb') as f:
    encoded_images = pickle.load(f)

# Create image array
ds = pd.read_csv("flickr_8k_train_dataset.txt", delimiter='\t').values
imgs = np.array([encoded_images[row[0]] for row in ds if row[0] in encoded_images])
print(f"Images Shape: {imgs.shape}")

# Limit to a subset for faster processing (adjust as needed)
number_of_images = 1500
captions = np.zeros([0, max_len])
next_words = np.zeros([0, vocab_size])
images = []
image_names = []

for ix in range(min(number_of_images, len(sentences))):
    captions = np.concatenate([captions, padded_sequences[ix:ix+1]])
    next_words = np.concatenate([next_words, subsequent_words[ix:ix+1]])
    for _ in range(len(padded_sequences[ix])):
        images.append(imgs[ix])
        image_names.append(ds[ix, 0])

captions = np.array(captions)
next_words = np.array(next_words)
images = np.array(images)
image_names = np.array(image_names)

# Save processed data
np.save("captions.npy", captions)
np.save("next_words.npy", next_words)
np.save("images.npy", images)
np.save("image_names.npy", image_names)

print(f"Final Captions Shape: {captions.shape}")
print(f"Final Next Words Shape: {next_words.shape}")
print(f"Final Images Shape: {images.shape}")
print(f"Final Image Names Length: {len(image_names)}")

# Save vocabulary for later use
with open("word_2_indices.p", "wb") as f:
    pickle.dump(word_2_indices, f)
with open("indices_2_word.p", "wb") as f:
    pickle.dump(indices_2_word, f)