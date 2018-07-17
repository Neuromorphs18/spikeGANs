import os
import numpy as np
import matplotlib.pyplot as plt


def get_frames(aedat, config):

    width = config.getint('input', 'width')
    height = config.getint('input', 'height')
    tau = eval(config.get('algorithm', 'tau'))

    num_frames = aedat['data']['frame']['numEvents']
    fps_updample_factor = config.getint('algorithm', 'fps_upsample_factor')
    num_frames_desired = num_frames * fps_updample_factor
    num_events = len(aedat['data']['polarity']['timeStamp'])
    num_events_per_frame = int(num_events / num_frames_desired)

    generator_image_path = config.get('paths', 'generator_image_path')
    target_image_path = config.get('paths', 'target_image_path')

    last_timestamp_array = np.zeros([width, height])
    last_polarity_array = np.zeros([width, height])

    frame_idx = 0
    for sub_frame_counter in range(num_frames_desired):

        if config.getboolean('algorithm', 'reset_timesurface'):
            last_timestamp_array = np.zeros_like(last_timestamp_array)
            last_polarity_array = np.zeros_like(last_polarity_array)

        for event_idx in range(sub_frame_counter * num_events_per_frame,
                               (sub_frame_counter + 1) * num_events_per_frame):

            x = aedat['data']['polarity']['x'][event_idx]
            y = aedat['data']['polarity']['y'][event_idx]
            p = aedat['data']['polarity']['polarity'][event_idx]
            t = aedat['data']['polarity']['timeStamp'][event_idx]

            last_timestamp_array[x, y] = t
            last_polarity_array[x, y] = 1 if p else -1

        last_timestamp = np.max(last_timestamp_array)

        time_surface = last_polarity_array * np.exp((last_timestamp_array -
                                                     last_timestamp) / tau)

        time_surface = np.fliplr(np.flipud(time_surface.transpose()))

        sub_frame_idx = sub_frame_counter % fps_updample_factor

        if sub_frame_counter > 0 and sub_frame_idx == 0:
            frame_idx += 1

        generator_file = os.path.join(
            generator_image_path, "{}.{}.png".format(frame_idx, sub_frame_idx))
        plt.imsave(generator_file, time_surface, cmap='jet')
        print('Saved sub-frame {} with last timestamp {:.0f}.'.format(
            sub_frame_counter, last_timestamp))

        if config.getboolean('output', 'save_aps_frames') \
                and sub_frame_idx == 0:
            aps_frame = aedat['data']['frame']['samples'][frame_idx]
            aps_frame = np.flipud(aps_frame)

            target_file = os.path.join(target_image_path,
                                       "{}.png".format(frame_idx))
            plt.imsave(target_file, aps_frame, cmap='gray')
