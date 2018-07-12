# spikeGANs

## spikes2Images
The purpose of this python notebook is to explore and create input images for GANs from the event-based and frame-based outputs of the DAVIS346 cameras. The code loads an aedat file (specified in the input_file_path) and writes the output images to a directory specified using the output_folder variable. Note that the code will create three sub-directories in the output folder and place images within those.

The code loads the data from the specified input file, extracts each frame and the dvs events occuring since the previous frame. These are then provided to the function *ConvertTDeventsToAccumulatedFrame*, which is responsible for converting the matrix of events into a 2D image for the GAN. The images are individually saved, and then concatenated horizontally and saved into the *combined* output folder. The combined images are suitable for the pix2pixs algorithm. The other two outputs are provided for use in other network types.

The data files recorded are very large and may take a very long time to load in Python. However, you can change the amount of data loaded from the AEDAT files using the inputparam member of the aedat dictionary. Edit the following lines to select a range of data, either by time or by event-index:

`#aedat['importParams']['startEvent'] = int(1e6)`

`#aedat['importParams']['endEvent'] = int(10e6)`

`#aedat['importParams']['startTime'] = 48`

`#aedat['importParams']['endTime'] = 49`

This same code will work with the DAVIS240C and any other DVS camera with a frame-based output, although the size of the image surface will need to be adjusted in the parameters to the function which creates the image representations from DVS data. It is currently configured for the DAVIS346 sensor.
