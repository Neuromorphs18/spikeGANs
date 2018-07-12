import numpy as np
from os import makedirs, listdir
from os.path import isfile, join, exists
from shutil import copyfile

# Define the inputs and outputs
input_folders = ["C:\\Data\\GAN Dataset\\Tobi\\ExponentialSurface\\combined"]
output_folder = 'C:\\Data\\GAN Dataset\\Tobi\\dataset'
splits = [80, 10, 10]
random_splits = False

# Check the splits and get a list of all the files in all the input directories
assert(sum(splits) == 100)
input_files = []
for folder_path in input_folders:
    assert(exists(folder_path))
    files = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    input_files.extend(files)

# Figure out the indexes for the train / test / validate sets
num_files = len(input_files)
training_size = int(np.floor((num_files) * (splits[0] / 100)))
validation_size = int(np.floor((num_files) * (splits[1] / 100)))
testing_size = num_files - (training_size + validation_size)
assert( (training_size + validation_size + testing_size) == num_files)

# Split the files into their relative splits
file_indexes = np.arange(1, len(input_files))
if random_splits:
    np.random.shuffle(file_indexes)
training_indexes = file_indexes[0:training_size]
validation_indexes = file_indexes[training_size:(training_size + validation_size)]
testing_indexes = file_indexes[(training_size + validation_size):]

# Retrieve the actual files to move using their index
training_files =  [input_files[i] for i in training_indexes]
validation_files = [input_files[i] for i in validation_indexes]
testing_files = [input_files[i] for i in testing_indexes]

# Create the output folders 
train_image_folder = join(output_folder, 'train')
test_image_folder = join(output_folder, 'test')
validate_image_folder = join(output_folder, 'val')
if not exists(train_image_folder):
    makedirs(train_image_folder)
if not exists(test_image_folder):
    makedirs(test_image_folder)
if not exists(validate_image_folder):
    makedirs(validate_image_folder)

# Copy the files across into the appropriate folders
for index, training_file in zip(range(0, len(training_files)), training_files):
    output_file_path = join(train_image_folder, "{}.png".format(index))
    copyfile(training_file,  output_file_path)
    print("Copying {} to {}".format(training_file, output_file_path))

for index, validation_file in zip(range(0, len(validation_files)), validation_files):
    output_file_path = join(validate_image_folder, "{}.png".format(index))
    copyfile(validation_file,  output_file_path)
    print("Copying {} to {}".format(validation_file, output_file_path))

for index, test_file in zip(range(0, len(testing_files)), testing_files):
    output_file_path = join(test_image_folder, "{}.png".format(index))
    copyfile(test_file,  output_file_path)
    print("Copying {} to {}".format(test_file, output_file_path))


print('Done!')