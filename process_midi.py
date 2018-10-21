import os
import sys
import numpy as np
import torch
import torch.utils.data as data

from data_preprocessing import *

# import midi processing and parsing scripts
sys.path.append('./midi/')

from utils import midiread, midiwrite

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def grab_data_path(genre, emotion):
	# grabs the data path of the datasets for training, validation, and testing data

	# Error Checker
	if genre not in ['Anime', 'Classical', 'Pop', 'Rock', 'Video_Game', 'Weeblyfe']:
		raise ValueError('Invalid genre category')
	if emotion not in ['Happy', 'Sad', 'Neutral']:
		raise ValueError('Invalid emotion category')

	path = './data/'

	train_path = path + genre + '/' + emotion.lower() + '_train/'
	test_path = path + genre + '/' + emotion.lower() + '_test/'
	val_path = path + genre + '/' + emotion.lower() + '_val/'

	return train_path, test_path, val_path


def midi_filename_piano_roll(midi_filename):
	midi_data = midiread(midi_filename, dt=0.3)

	piano_roll = midi_data.piano_roll.T

	# Binarize pressed notes
	piano_roll[piano_roll > 0] = 1

	return piano_roll


def pad_piano_roll(piano_roll, max_length=132333, pad_value=0):
	# Pad 0 at beginning of sequence

	# Hardcode 88 since 88 pitches are always used
	original_piano_roll_length = piano_roll.shape[1]

	padded_piano_roll = np.zeros((88, max_length))
	padded_piano_roll[:] = pad_value

	padded_piano_roll[:, :original_piano_roll_length] = piano_roll

	return padded_piano_roll


class NotesGenerationDataset(data.Dataset):
	def __init__(self, midi_folder_path, longest_sequence_length=1491):

		self.midi_folder_path = midi_folder_path

		midi_filenames = os.listdir(midi_folder_path)

		self.longest_sequence_length = longest_sequence_length

		midi_full_filenames = [os.path.join(midi_folder_path, filename) for filename in midi_filenames]

		self.midi_full_filenames = list(midi_full_filenames)

		if longest_sequence_length is None:
			self.update_the_max_length()


	def update_the_max_length(self):
		"""
		Recomputes longest seq const in dataset. Reads all midi files and finds max length
		"""

		sequences_lengths = [midi_filename_piano_roll(filename).shape[1] for filename in self.midi_full_filenames]

		max_length = max(sequences_lengths)

		self.longest_sequence_length = max_length


	def __len__(self):
		return len(self.midi_full_filenames)

	def __getitem__(self, index):

		midi_full_filename = self.midi_full_filenames[index]

		piano_roll = midi_filename_piano_roll(midi_full_filename)

		# -1 shift
		sequence_length = piano_roll.shape[1] - 1

		# Median length of songs
		if piano_roll.shape[1] > self.longest_sequence_length:
			random_strt_idx = np.random.randint(0, piano_roll.shape[1] - 
				                                self.longest_sequence_length+3, 1)
			ground_truth_sequence = piano_roll[:, random_strt_idx[0]:random_strt_idx[0]+self.longest_sequence_length]
			input_truth_sequence = piano_roll[:, random_strt_idx[0]-1:random_strt_idx[0]+self.longest_sequence_length]
		else:
			input_sequence = piano_roll[:, :-1]
			ground_truth_sequence = piano_roll[:, 1:]
		# pad such that all seq have same length
		input_sequence_padded = pad_piano_roll(input_sequence, max_length=self.longest_sequence_length)

		ground_truth_sequence_padded = pad_piano_roll(ground_truth_sequence,
						                              max_length=self.longest_sequence_length,
						                              pad_value=0)
		input_sequence_padded = input_sequence_padded.transpose()
		ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()

		return (torch.FloatTensor(input_sequence_padded),
			    torch.LongTensor(ground_truth_sequence_padded),
			    torch.LongTensor([self.longest_sequence_length]))


class Dataset_validation(data.Dataset):
	def __init__(self, val_list, longest_sequence_length=1491):
		self.val_list = val_list
		self.longest_sequence_length = longest_sequence_length

	def __len__(self):
		return len(self.val_list)

	def __getitem__(self):
		piano_roll = self.val_list[index]
		input_sequence = piano_roll[:, :-1]
		ground_truth_sequence = piano_roll[:, 1:]

		input_sequence_padded = pad_piano_roll(input_sequence, max_length=self.longest_sequence_length)

		ground_truth_sequence_padded = pad_piano_roll(ground_truth_sequence,
						                              max_length=self.longest_sequence_length,
						                              pad_value=0)

		input_sequence_padded = input_sequence_padded.transpose()
		ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()

		return (torch.FloatTensor(input_sequence_padded),
			    torch.LongTensor(ground_truth_sequence_padded),
			    torch.LongTensor([sequene_length]))


def post_process_sequence_batch(batch_tuple):
	input_sequences, output_sequences, lengths = batch_tuple

	splitted_input_sequence_batch = input_sequences.split(split_size=1)
	splitted_output_sequence_batch = output_sequences.split(split_size=1)
	splitted_lengths_batch = lengths.split(split_size=1)

	training_data_tuples = zip(splitted_input_sequence_batch,
		                       splitted_output_sequence_batch,
		                       splitted_lengths_batch)

	training_data_tuples_sorted = sorted(training_data_tuples,
									     key=lambda p: int(p[2]),
									     reverse=True)

	splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(*training_data_tuples_sorted)

	input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
	output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
	lengths_batch_sorted = torch.cat(splitted_lengths_batch)

	# trim overall data matrix using size of longest sequence
	input_sequence_batch_sorted = input_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0], :]
	output_sequence_batch_sorted = output_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0], :]

	input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)

	# pytorch api needs lengths to be list of ints
	lengths_batch_sorted_list = list(lengths_batch_sorted)
	lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)

	return input_sequence_batch_transposed, output_sequence_batch_sorted, list(lengths_batch_sorted_list)


# test
if __name__ == "__main__":
	anime = fn('Anime', 'Happy')
	happy, sad, neutral = extract_files('Anime')
	tr, va, te = create_dataset(happy)
	save_dataset(tr, va, te, 'Anime', 'Happy')
	train_path, test_path, val_path = grab_data_path('Anime', 'Happy')
	print(train_path)
	trainset = NotesGenerationDataset(train_path)
	trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=20,
		                                          shuffle=True, num_workers=0, drop_last=True)
	
	print(len(trainset), len(trainset_loader))
	X = next(iter(trainset_loader))
	print(X[0].shape)
	print(X[1].shape)
	print(torch.sum(X[0]))
	
