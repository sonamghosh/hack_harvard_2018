import os
import sys
import random
import shutil
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


# Save files into their respective directory
def save_dataset(train_set, val_set, test_set, genre, emotion):
	path = './data/'+genre+'/'+emotion+'/'
	test_path = './data/'+genre+ '/'+emotion.lower()+'_test/'
	train_path = './data/'+genre+'/'+emotion.lower()+'_train/'
	val_path = './data/'+genre+'/'+emotion.lower()+'_val/'

	for f in train_set:
		shutil.copy(path+f, train_path)
	for f in test_set:
		shutil.copy(path+f, test_path)
	for f in val_set:
		shutil.copy(path+f, val_path)


# Delete files in Train, Val, Test
def delete_dataset(genre, emotion):
	train_path = './data/'+genre+'/'+emotion.lower()+'_train/'
	test_path = './data/'+genre+'/'+emotion.lower()+'_test/'
	val_path = './data/'+genre+'/'+emotion.lower()+'_val/'

	train_list = os.listdir(train_path)
	test_list = os.listdir(test_path)
	val_list = os.listdir(val_path)

	for filename in train_list:
		if filename.endswith('.mid'):
			os.unlink(train_path+filename)
	for filename in test_list:
		if filename.endswith('.mid'):
			os.unlink(test_path+filename)
	for filename in val_list:
		if filename.endswith('.mid'):
			os.unlink(val_path+filename)




# Test
if __name__ == "__main__":
	a = fn('Anime', 'Happy')
	b, c, d = extract_files('Anime')
	tr, va, te = create_dataset(b)  # Happy
	save_dataset(tr, va, te, 'Anime', 'Happy')
	#delete_dataset('Anime', 'Happy')

