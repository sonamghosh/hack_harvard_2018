import os
import sys
import random

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

"""
# Create training, testing, and validation datasets
def create_dataset(data, train_split, test_split, valid_test)
"""




# Test
if __name__ == "__main__":
	a = fn('Anime', 'Happy')
	b, c, d = extract_files('Anime', 50)