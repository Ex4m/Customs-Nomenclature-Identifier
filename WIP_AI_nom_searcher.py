import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import time

def train_model(descriptions, hs_codes, epochs):
    # Tokenize descriptions
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(descriptions)
    descriptions = tokenizer.texts_to_matrix(descriptions)

    # Create a LabelEncoder object
    le = LabelEncoder()

    # Fit the LabelEncoder on the hs_codes
    le.fit(hs_codes)

    # Transform the hs_codes into numerical values
    hs_codes = le.transform(hs_codes)

    # One-Hot Encoding of the labels
    hs_codes = np_utils.to_categorical(hs_codes)

    # Build the neural network
    model = Sequential()
    model.add(Dense(64, input_shape=(5000,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(hs_codes.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(descriptions, hs_codes, epochs=epochs, batch_size=16)
    
    # Save the tokenizer and model
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.save("model.tflearn.Nom")
    return tokenizer, model


trained_data = pd.read_json("Nom_output", lines = True)
trained_data = pd.DataFrame(trained_data)
try:
    loaded_model = load_model("model.tflearn.Nom")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as chyba:
    print(chyba)
    epochs = int(input("How many epochs to train? "))
    # Extract input and output sets from data
    descriptions = trained_data['Description'].values
    hs_codes = trained_data['HS Code'].astype(int)

    tokenizer, model = train_model(descriptions, hs_codes, epochs)

# Make a prediction
while True:
    user_input = input("Enter a Description: ")
    if tokenizer:
        user_input = tokenizer.texts_to_matrix([user_input])
    else:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        user_input = tokenizer.texts_to_matrix([user_input])
    hs_code = np.argmax(loaded_model.predict(user_input))
    data_filtered = trained_data.loc[trained_data['HS Code'] == hs_code]
    if data_filtered.empty:
        print("Predicted HS Code is not in the dataset.")
        print(hs_code)
        time.sleep(2)
        print(trained_data['HS Code'])
    else:
        description = data_filtered['Description'].values[0]
        print("Predicted HS Code: ", hs_code)
        print("Predicted Description: ", description)
