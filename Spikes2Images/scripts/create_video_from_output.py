import numpy as np
import cv2
from PIL import Image
from os import makedirs, listdir
from os.path import isfile, join, exists

folder_path = "C:\\Users\\gregc\\Google Drive\\Research\\Projects\\Project Spikes2Gan\\pytorch-CycleGAN-and-pix2pix\\results\\elisa1\\test_latest\\images"

files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

spikes = [join(folder_path, f) for f in files if f.endswith('_real_A.png')] 
fake = [join(folder_path, f) for f in files if f.endswith('_fake_B.png')]
real = [join(folder_path, f) for f in files if f.endswith('_real_B.png')]

num_frames = len(spikes)
assert(len(spikes) == len(fake))
assert(len(spikes) == len(real))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('video.avi',fourcc, 10.0, (512, 256))

for frame_counter in range(0, num_frames):
    print('Generating frame {} of {}...'.format(frame_counter, num_frames))
    img_a = Image.open(spikes[frame_counter])
    img_b = Image.open(fake[frame_counter])
    assert(img_a.size == img_b.size)
    aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
    aligned_image.paste(img_a, (0, 0))
    aligned_image.paste(img_b, (img_a.size[0], 0))

    frame = np.array(aligned_image.convert('RGB'))

    cv2.imshow('frame',frame)

    out.write(frame)

out.release()
cv2.destroyAllWindows()

print('Done!')
"""
# Define the codec and create VideoWriter object



# write the flipped frame
#        


# Release everything if job is finished
cv2.destroyAllWindows()

"""