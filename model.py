import pretty_midi
import numpy as np 
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.utils.data as data
import os
import random
import sys
import matplotlib.pyplot as plt 
import skimage.io as io 
from pathlib import Path 

# import midi
sys.path.append('./midi/')

from utils import midiwrite, midiread 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from data_preprocessing import *
from process_midi import *


# Stacked Long Short-Time Memory Recurrent Neural Network (LSTM-RNN)

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, n_layers=2):
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.n_layers = n_layers

		self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)

		self.bn = nn.BatchNorm1d(hidden_size)

		self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=0.2)

		self.logits_fc = nn.Linear(hidden_size, num_classes)


	def forward(self, input_sequences, input_sequences_lengths, hidden=None):
		batch_size = input_sequences.shape[1]

		notes_encoded = self.notes_encoder(input_sequences)
		notes_encoded_rolled = notes_encoded.permute(1,2,0).contiguous()
		notes_encoded_norm = self.bn(notes_encoded_rolled)
		notes_encoded_norm_drop = nn.Dropout(0.25)(notes_encoded_norm)
		notes_encoded_complete = notes_encoded_norm_drop.permute(2,0,1)

		# Run on non padded regions of batch
		packed = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded_complete, input_sequences_lengths)
		outputs, hidden = self.lstm(packed, hidden)
		outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

		outputs_norm = self.bn(outputs.permute(1,2,0).contiguous())
		outputs_drop = nn.Dropout(0.1)(outputs_norm)
		logits = self.logits_fc(outputs_drop.permute(2,0,1))
		logits = logits.transpose(0, 1).contiguous()

		neg_logits = (1 - logits)

		# Cross Entropy Loss
		binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
		logits_flatten = binary_logits.view(-1, 2)
		return logits_flatten, hidden 



def validate(model):
	model.eval()
	full_val_loss = 0.0
	overall_sequence_length = 0.0

	for batch in valset_loader:
		if torch.cuda_is_available():
			post_processed_batch_tuple = post_processed_sequence_batch(batch)

			input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

			output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1).cuda())

			input_sequences_batch_var = Variable(input_sequences_batch.cuda())

			logits, _ = model(input_sequences_batch_var, sequences_lengths)

			loss = criterion_val(logits, output_sequences_batch_var)

			full_val_loss += loss.item()

			overall_sequence_length += sum(sequences_lengths)
		else:
			post_processed_batch_tuple = post_processed_sequence_batch(batch)

			input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

			output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1))

			input_sequences_batch_var = Variable(input_sequences_batch)

			logits, _ = model(input_sequences_batch_var, sequences_lengths)

			loss = criterion_val(logits, output_sequences_batch_var)

			full_val_loss += loss.item()

			overall_sequence_length += sum(sequences_lengths)

	return full_val_loss / (overall_sequence_length * 88)

