# Import the required libraries to load and execute the code
from os import makedirs
from os.path import join, exists
import argparse

from PIL import Image

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from PyAedatTools.ImportAedat import ImportAedat
from PyAedatTools.ImportAedatHeaders import ImportAedatHeaders
from PyAedatTools.ImportAedatDataVersion1or2 import ImportAedatDataVersion1or2


def ConvertTDeventsToAccumulatedFrame(state, td_events, width=346, height=260, tau=1e6):
    """ Converts the specified TD data to a frame by accumulating events """
    if state is None:
        state = {}
        state['last_spike_time'] = np.zeros([width, height])
        state['last_spike_polarity'] = np.zeros([width, height])
    td_image = np.zeros([width, height])
    
    last_t = 0
    for row in td_events:
        [x, y, p, t] = row
    
        state['last_spike_time'][x, y] = t
        state['last_spike_polarity'][x, y] = 1 if p == True else -1
        last_t = t 
        
        td_image[x,y] = p
    
    surface_image = state['last_spike_polarity']*np.exp((state['last_spike_time'] - last_t) / tau)
    
    #return [state, td_image.transpose()]
    return [state, surface_image.transpose()]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input aedat')
parser.add_argument('-o', '--output', help='output folder')
parser.add_argument('-t', '--tau', default=1e6, help='time constant')
parser.add_argument('-n', '--name', help='output file prefix')
args = parser.parse_args()

# Configure the input and output folders
input_file_path = args.input
output_folder = args.output
tau = args.tau

generator_image_folder = join(output_folder, 'generator')
target_image_folder = join(output_folder, 'targets')
combined_image_folder = join(output_folder, 'combined')
if not exists(generator_image_folder):
    makedirs(generator_image_folder)
if not exists(target_image_folder):
    makedirs(target_image_folder)
if not exists(combined_image_folder):
    makedirs(combined_image_folder)

# Configure the reading parameters
aedat = {}
aedat['importParams'] = {}
aedat['importParams']['filePath'] = input_file_path

# Configure which parts of the file to read
#aedat['importParams']['startEvent'] = int(1e6);
#aedat['importParams']['endEvent'] = int(10e6);
#aedat['importParams']['startTime'] = 48;
#aedat['importParams']['endTime'] = 49;

aedat['importParams']['dataTypes'] = {'polarity', 'special', 'frame'};

# Invoke the function
aedat = ImportAedat(aedat)
print('Read {} seconds of data'.format((aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp']) / 1e6))
print('Read {} events.'.format(aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp']))
print('Done!')

num_frames = aedat['data']['frame']['numEvents']
num_events = aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp']
assert(num_frames > 0)
assert(num_frames == len(aedat['data']['frame']['samples']))

last_frame_start = aedat['data']['frame']['timeStampStart'][0]
last_frame_end = aedat['data']['frame']['timeStampEnd'][0]
last_td_timestamp_index = 0
last_td_timestamp = aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]

update_interval = round(max(num_frames / 100, 1))
state = None

# Loop through the frames and save them accordingly
for frame_counter in range(1, num_frames):
    
    # Find out when the frame started and ended
    frame_start = aedat['data']['frame']['timeStampStart'][frame_counter]
    frame_end = aedat['data']['frame']['timeStampEnd'][frame_counter]
    
    # Figure out the range of TD times that we are interested in
    assert(last_frame_end <= frame_start)
    td_range = [last_frame_end, frame_start]
    last_frame_start = aedat['data']['frame']['timeStampStart'][frame_counter]
    last_frame_end = aedat['data']['frame']['timeStampEnd'][frame_counter]    
    
    # Extract the TD region for this frame
    assert(last_td_timestamp <= td_range[0]) # Ensure that we're seeking to a time in the future
    # Find the starting event
    while (last_td_timestamp < td_range[0] and last_td_timestamp_index < num_events):
        last_td_timestamp_index = last_td_timestamp_index + 1
        last_td_timestamp = aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]
    # Now find the last event and store all intermediate events in the array
    td_events = []
    while (last_td_timestamp < td_range[1] and last_td_timestamp_index < num_events):
        td_events.append([aedat['data']['polarity']['x'][last_td_timestamp_index],
                          aedat['data']['polarity']['y'][last_td_timestamp_index],
                          aedat['data']['polarity']['polarity'][last_td_timestamp_index],
                          aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]])
        last_td_timestamp_index = last_td_timestamp_index + 1
        last_td_timestamp = aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]
    # Convert the events to a numpy array
    td_events = np.array(td_events)
    print('Frame: {} Read {} events from time {} to time {}'.format(frame_counter, td_events.shape[0], td_range[0], td_range[1]))
    
    
    # Extract the frame data and flip it accordingly
    aps_frame = aedat['data']['frame']['samples'][frame_counter]
    state, td_frame = ConvertTDeventsToAccumulatedFrame(state, td_events, tau=tau);
    td_frame = np.fliplr(np.flipud(td_frame))
    aps_frame = np.flipud(aps_frame)
    
    # Save the images
    generator_file = join(generator_image_folder, "{}.png".format(frame_counter))
    target_file = join(target_image_folder, "{}.png".format(frame_counter))
    cmap = plt.cm.jet
    plt.imsave(generator_file, td_frame, cmap=cmap)
    plt.imsave(target_file, aps_frame, cmap='gray')
    
    # Concatenate the two images together
    img_a = Image.open(generator_file)
    img_b = Image.open(target_file)
    if (img_a.size == img_b.size):
        #assert(img_a.size == img_b.size)
        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        combined_file = join(combined_image_folder, args.name+"-{:04d}.png".format(frame_counter))
        aligned_image.save(combined_file)    
    
matplotlib.pyplot.figure()
matplotlib.pyplot.subplot(1,2,1)
matplotlib.pyplot.imshow(aps_frame)
matplotlib.pyplot.title('Frame: {} Read {} events from time {} to time {}'.format(frame_counter, td_events.shape[0], td_range[0], td_range[1]))
matplotlib.pyplot.subplot(1,2,2)
matplotlib.pyplot.imshow(td_frame)
