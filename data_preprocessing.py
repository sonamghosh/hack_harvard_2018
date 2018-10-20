import os
import sys
import random
import pdb

# This Script handles creating Training, Test, and Validation datasets


# List files in directory
def fn(genre, emotion):
	# Error Checkers
	if genre not in ['Anime', 'Classical', 'Pop', 'Rock', 'Video_Game']:
		raise ValueError('Invalid genre category')

	if emotion not in ['Happy', 'Sad', 'Neutral']:
		raise ValueError('Invalid emotion dataset')

	file_list = os.listdir('./data/'+genre+'/'+emotion+'/')
	
	return file_list


# Randomly choose N number of files from each
def extract_files(genre, num_files=50):
	happy_list = fn(genre, 'Happy')
	sad_list = fn(genre, 'Sad')
	neutral_list = fn(genre, 'Neutral')

	# Error Checker
	if any(len(lst) < num_files for lst in [happy_list, sad_list, neutral_list]):
		raise ValueError('There must be atleast ' + str(num_files) + ' files in each of the emotion datasets')


	# Sample through and pick N number of files
	happy_list = random.sample(happy_list, num_files)
	sad_list = random.sample(sad_list, num_files)
	neutral_list = random.sample(neutral_list, num_files)

	return happy_list, sad_list, neutral_list


# Create training, testing, and validation datasets
def create_dataset(data, train_split=0.8, test_split=0.2):
	# Error Checker
	if train_split + test_split != 1.:
		raise ValueError('The split ratio must add up to 1.0')
	# Shuffle filenames
	random.shuffle(data)
	# Split Train - Test 
	split_1 = int(train_split * len(data))
	split_2 = int((1 - test_split/2) * len(data))

	train_set = data[:split_1]
	val_set = data[split_1:split_2]
	test_set = data[split_2:]

	return train_set, val_set, test_set










# Test
if __name__ == "__main__":
	a = fn('Anime', 'Happy')
	b, c, d = extract_files('Anime', 50)
	print(b)
	tr, val, te = create_dataset(b)
	print(len(tr))
	print(len(val))
	print(len(te))

