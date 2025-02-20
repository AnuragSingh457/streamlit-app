import tensorflow as tf
keras = tf.keras
models = tf.keras.models

from keras._tf_keras.keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.utils import load_img, img_to_array
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load Keras model
model = load_model("image_captioning_model.keras")

# Load pre-trained InceptionV3 model
inception = InceptionV3(weights="imagenet")
inception = tf.keras.Model(inception.input, inception.layers[-2].output)  

def preprocess_image(img):
    if isinstance(img, str):  
        img = load_img(img, target_size=(299, 299))
    elif isinstance(img, Image.Image):  
        img = img.resize((299, 299))
    else:
        raise ValueError("Invalid image input")

    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = inception.predict(img)
    
    if features.shape[-1] != 2048:
        raise ValueError(f"Expected features shape (1, 2048), got {features.shape}")
    
    return features

def remove_special_tokens(caption):
    """Remove special tokens and clean the caption."""
    special_tokens = {'<start>', '<end>', '<pad>', '<unk>'}
    words = caption.split()
    cleaned_words = [word for word in words if word not in special_tokens and 'end' not in word.lower()]
    return ' '.join(cleaned_words)

def generate_caption(image_file):
    # Process the image
    image_pil = Image.open(image_file).convert("RGB")
    image_pil.save("temp.jpg")
    image_tensor = preprocess_image("temp.jpg")
    
    # Initialize sequence with start token
    start_token_id = tokenizer.texts_to_sequences(['<start>'])[0][0]
    current_sequence = [start_token_id]
    
    # Set up generation parameters
    max_length = 34
    end_token_id = tokenizer.texts_to_sequences(['<end>'])[0][0]
    
    # Generate the caption word by word
    generated_words = []
    
    for _ in range(max_length):
        # Pad the sequence
        padded_sequence = pad_sequences([current_sequence], maxlen=max_length, padding='post')
        
        # Get predictions
        predictions = model.predict([image_tensor, padded_sequence], verbose=0)[0]
        
        # Get the predicted word ID
        predicted_id = np.argmax(predictions)
        
        # Stop if we predict the end token
        if predicted_id == end_token_id:
            break
        
        # Get the actual word
        predicted_word = tokenizer.index_word.get(predicted_id, '')
        
        # Only add the word if it's not a special token
        if predicted_word and predicted_word not in {'<start>', '<end>', '<pad>', '<unk>'}:
            generated_words.append(predicted_word)
        
        # Update the sequence for next prediction
        current_sequence.append(predicted_id)
    
    # Join the words and clean up any remaining special tokens
    caption = ' '.join(generated_words)
    final_caption = remove_special_tokens(caption)
    
    return final_caption

st.title("Image Captioning App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    caption = generate_caption(uploaded_file)
    st.write(f"Generated Caption: {caption}")