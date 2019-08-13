from flask import Flask, request, jsonify, render_template

from torch import optim
import unicodedata
import os
import re
import codecs
import torch
import math
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from greedy import GreedySearchDecoder
from encoder import EncoderRNN
from decoder import LuongAttnDecoderRNN
from flask_socketio import SocketIO
from voc import Voc
# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

# We establish a seed for the replication of the experiments correctly
seed = 0
torch.manual_seed(seed=seed)

@app.route('/')
def sessions():
    return render_template('session.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

#TODO: write a method to take in user input from the client, 
#send it to the bot and display the results.

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500


# Configure models
MAX_LENGTH = 10
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = "6000_checkpoint.tar"
checkpoint_iter = 6000

voc = Voc("cornell movie-dialogs corpus")
# If loading on same machine the model was trained on
if loadFilename:
    #TODO

print('Building encoder and decoder ...')
#TODO

print('Models built and ready to go!')

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def unicodeToAscii(s):
    return ''.join(
        # remove accentuated words and other things like that
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters using regex
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to('cpu')
    lengths = lengths.to('cpu')
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluateInput(input_sentence, encoder, decoder, searcher, voc):
    print("input sentence in evaluateInput:", input_sentence)
    while(1):
        #TODO

# @app.route('/predict', methods=['POST'])
def predict(input_sentence):
    # Set dropout layers to eval mode
    # Initialize search module
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    return evaluateInput(input_sentence, encoder, decoder, searcher, voc)

if __name__ == "__main__":
    socketio.run(app, debug=True)
