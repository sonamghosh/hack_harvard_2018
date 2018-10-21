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
from model import *

clip = 1.0
epochs_number = 10
sample_history = []
best_val_loss = float("inf")

# Training 

def lrfinder(start, end, model, trainset_loader, epochs=10):
	model.train()
	lrs = np.linspace(start, end, epochs*len(trainset_loader))
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters, start)
	loss_lost = []
	ctr = 0

	for epoch_number in range(epochs):
		epoch_loss = []
		for batch in trainset_loader:
			optimizer.param_groups[0]['lr'] = lrs[ctr]
			ctr = ctr + 1

			post_processed_batch_tuple = post_process_sequence_batch(batch)

			input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

			if torch.cuda.is_available():
				output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1).cuda())
				input_sequences_batch_var = Variable(input_sequences_batch.cuda())

				optimizer.zero_grad()

				logits, _ = model(input_sequences_batch_var, sequences_lengths)

				loss = criterion(logits, output_sequences_batch_var)
				loss_list.append(loss.item())
				loss.backward()

				torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)

				optimizer.step()
			else:
				output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1))
				input_sequences_batch_var = Variable(input_sequences_batch)

				optimizer.zero_grad()

				logits, _ = model(input_sequences_batch_var, sequences_lengths)

				loss = criterion(logits, output_sequences_batch_var)
				loss_list.append(loss.item())
				loss.backward()

				torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)

				optimizer.step()

		if epoch_number%5 == 0:
			print('Epoch %d' % epoch_number)
	plt.plot(lrs, loss_list)
	return lrs, loss_list


def get_triangular_lr(lr_low, lr_high, mini_batches, epochs_number=1):
	iterations = mini_batches*epochs_number
	lr_mid = lr_high/7 + lr_low
	up = np.linspace(lr_low, lr_high, int(round(iterations*0.35)))
	down = np.linspace(lr_high, lr_mid, int(round(iterations*0.35)))
	floor = np.linspace(lr_mid, lr_low, int(round(iterations*0.30)))

	return np.hstack([up, down[1:], floor])


criterion = nn.CrossEntropyLoss()
criterion_val = nn.CrossEntropyLoss(size_average=False)

def train_model(model, trainset_loader, lrs_triangular, epochs_number=10, wd=0.0, best_val_loss=float("inf")):
	loss_list = []
	val_list = []
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = torch.optim.Adam(parameters, lr=lrs_triangular[0], weight_decay=wd)
	ctr = 0

	for epoch_number in range(epochs_number):
		model.train()
		epoch_loss = []
		for batch in trainset_loader:
			try:
				optimizer.param_groups[0]['lr'] = lrs_triangular[ctr]
			except IndexError: pass
			ctr+=1
			if torch.cuda.is_available():
				post_processed_batch_tuple = post_process_sequence_batch(batch)
				input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
				output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1).cuda())
				input_sequences_batch_var = Variable(input_sequences_batch.cuda())

				optimizer.zero_grad()

				logits, _ = model(input_sequences_batch_var, sequences_lengths)

				loss = criterion(logits, output_sequences_batch_var)
				loss_list.append(loss.item())
				epoch_loss.append(loss.item())
				loss.backward()

				torch.nn.utils.clip_grad_norm(model.parameters(), clip)

				optimizer.step()
			else:
				post_processed_batch_tuple = post_process_sequence_batch(batch)
				input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
				output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1))
				input_sequences_batch_var = Variable(input_sequences_batch)

				optimizer.zero_grad()

				logits, _ = model(input_sequences_batch_var, sequences_lengths)

				loss = criterion(logits, output_sequences_batch_var)
				loss_list.append(loss.item())
				epoch_loss.append(loss.item())
				loss.backward()

				torch.nn.utils.clip_grad_norm(model.parameters(), clip)

				optimizer.step()

		current_trn_epoch = sum(epoch_loss)/len(trainset_loader)
		current_val_loss = validate(model)

		print('Training Loss: Epoch:', epoch_number, ':', current_trn_epoch)
		print('Validation Loss: Epoch', epoch_number, ':', current_val_loss)
		print('')

		val_list.append(current_val_loss)

		if current_val_loss < best_val_loss:
			torch.save(model.state_dict(), 'music_anime_new.pth')
			best_val_loss = current_val_loss
	return best_val_loss

