import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
import pandas as pd

# Load the dataset
df = pd.read_json("Nom_output", lines= True)

# Extract the descriptions and hs codes
descriptions = df["Description"].tolist()
hs_codes = df["HS Code"].tolist()

# Tokenize the descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions)
word_index = tokenizer.word_index

# Convert descriptions to sequences
sequences = tokenizer.texts_to_sequences(descriptions)

# Pad sequences to ensure they have the same length
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Create the model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=64, input_length=max_length),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model
np_paddseq = np.array(padded_sequences).astype(int)
n_padrows = np_paddseq.shape[0] 
np_paddseq = np_paddseq.reshape((n_padrows, max_length))
np_hscodes = np.array(hs_codes)
n_rows = np_hscodes.shape[0] 
np_hscodes = np_hscodes.reshape((n_rows,1))
epochs = int(input("How many epochs? "))
model.fit(np_paddseq, np_hscodes, epochs=epochs)

# Predict on a sample input
sample_input = " research"
sample_input_sequence = tokenizer.texts_to_sequences([sample_input])
padded_sample_input = pad_sequences(sample_input_sequence, maxlen=max_length, padding="post")
# Predict on all inputs
padded_inputs = np.array(padded_sequences)
predictions = model.predict(padded_inputs)

# Find the index of the highest predicted probability for each input
best_match_indices = np.argmax(predictions, axis=1)

# Output the best match for each input
print("Best matches:")
for i in range(len(descriptions)):
    
  print(f"{descriptions[i]} -> {descriptions[best_match_indices[i]]} (HS Code: {np_hscodes[best_match_indices[i]]})")


















#2nd type of training
"""
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import pad_sequences
from keras.utils import to_categorical
import pickle
import time
import sys

# Load the dataset
df = pd.read_json("Nom_output", lines= True)

# Extract the descriptions and hs codes
descriptions = df["Description"].tolist()
hs_codes = df["HS Code"].tolist()


# Create a tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(descriptions)
sequences = tokenizer.texts_to_sequences(descriptions)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Define the maximum sequence length
max_length = max([len(s) for s in sequences])
# Pad the sequences to the same length
padded_sequences = pad_sequences(sequences, maxlen=max_length)
padded_sequences = [list(map(int,sequence)) for sequence in padded_sequences]

# Create the model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 300, input_length=max_length))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=len(set(hs_codes)), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the number of training epochs
epochs = int(input("How much epochs to train? "))

# Fit the model
np_hscodes = np.array(hs_codes).astype(int)
n_rows = np_hscodes.shape[0] 
np_hscodes = np_hscodes.reshape(n_rows,1).astype(int)
np_paddseq = np.array(padded_sequences).astype(int)
 
try:
    model.fit(np_paddseq, np_hscodes, epochs=epochs)
except Exception as e:
    print("Error encountered during model fitting:")
    print(e)
    print("Location of the error:", sys.exc_info()[2].tb_lineno)
    print("Data causing the error:")
    print("np_paddseq:", np_paddseq)
    print("np_hscodes:", np_hscodes)

# Save the model
model.save("hs_code_model.h5")
print("model saved")"""


















"""
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
"""
