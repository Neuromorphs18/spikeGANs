import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def get_event_surface_from_events(td_events, config, state):
    """ Converts the specified TD data to an time surface frame."""

    width = config.getint('input', 'width')
    height = config.getint('input', 'height')
    tau = eval(config.get('algorithm', 'tau'))

    if state is None:
        state = {'last_spike_time': np.zeros([width, height]),
                 'last_spike_polarity': np.zeros([width, height])}

    td_image = np.zeros([width, height])

    last_t = 0
    for event in td_events:
        [x, y, p, t] = event

        state['last_spike_time'][x, y] = t
        state['last_spike_polarity'][x, y] = 1 if p else -1
        last_t = t

        td_image[x, y] = p

    surface_image = state['last_spike_polarity'] * np.exp(
        (state['last_spike_time'] - last_t) / tau)

    return [state, surface_image.transpose()]


def get_frames(aedat, config):
    num_frames = aedat['data']['frame']['numEvents']
    num_events = aedat['info']['lastTimeStamp'] - \
        aedat['info']['firstTimeStamp']
    assert (num_frames > 0)
    assert (num_frames == len(aedat['data']['frame']['samples']))

    last_frame_start = aedat['data']['frame']['timeStampStart'][0]
    last_frame_end = aedat['data']['frame']['timeStampEnd'][0]
    last_td_timestamp_index = 0
    last_td_timestamp = \
        aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]

    update_interval = round(max(num_frames / 100, 1))

    event_data = []
    event_id = 0

    # Loop through the frames and save them accordingly
    for frame_counter in range(1, num_frames):

        # Find out when the frame started and ended
        frame_start = aedat['data']['frame']['timeStampStart'][frame_counter]
        frame_end = aedat['data']['frame']['timeStampEnd'][frame_counter]

        # Figure out the range of TD times that we are interested in
        assert (last_frame_end <= frame_start)
        td_range = [last_frame_end, frame_start]
        last_frame_start = \
            aedat['data']['frame']['timeStampStart'][frame_counter]
        last_frame_end = aedat['data']['frame']['timeStampEnd'][frame_counter]

        # Extract the TD region for this frame

        # Ensure that we're seeking to a time in the future
        assert (last_td_timestamp <= td_range[0])
        # Find the starting event
        while last_td_timestamp < td_range[0] and \
                last_td_timestamp_index < num_events:
            last_td_timestamp_index += 1
            last_td_timestamp = \
                aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]
        # Now find the last event and store all intermediate events in the
        # array

        td_events = []
        frame_now = 1
        while last_td_timestamp < td_range[1] and \
                last_td_timestamp_index < num_events:
            td_events.append(
                [aedat['data']['polarity']['x'][last_td_timestamp_index],
                 aedat['data']['polarity']['y'][last_td_timestamp_index],
                 aedat['data']['polarity']['polarity'][last_td_timestamp_index],
                 aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]])
            event_id += 1
            frame_now = 0
            last_td_timestamp_index += 1
            last_td_timestamp = \
                aedat['data']['polarity']['timeStamp'][last_td_timestamp_index]
        # Convert the events to a numpy array
        td_events = np.array(td_events)

        event_data += td_events

        print('Frame: {} Read {} events from time {} to time {}'.format(
            frame_counter, td_events.shape[0], td_range[0], td_range[1]))

        state = None

        # Extract the frame data and flip it accordingly
        state, td_frame = get_event_surface_from_events(td_events, config,
                                                        state)
        aps_frame = aedat['data']['frame']['samples'][frame_counter]
        td_frame = np.fliplr(np.flipud(td_frame))
        aps_frame = np.flipud(aps_frame)

        # Save the images
        generator_file = os.path.join(config.get(
            'paths', 'generator_image_path'), "{}.png".format(frame_counter))
        target_file = os.path.join(config.get('paths', 'target_image_path'),
                                   "{}.png".format(frame_counter))
        cmap = plt.cm.jet
        plt.imsave(generator_file, td_frame, cmap=cmap)
        plt.imsave(target_file, aps_frame, cmap='gray')

        # Concatenate the two images together
        img_a = Image.open(generator_file)
        img_b = Image.open(target_file)
        assert (img_a.size == img_b.size)
        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        combined_file =\
            os.path.join(config.get('paths', 'combined_image_path'),
                         "{:04d}.png".format(frame_counter))
        aligned_image.save(combined_file)

        if frame_counter % update_interval == 0:
            # Render the frame
            # plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(aps_frame)
            plt.title(
                'Frame: {} Read {} events from time {} to time {}'.format(
                    frame_counter, td_events.shape[0], td_range[0],
                    td_range[1]))
            plt.subplot(1, 2, 2)
            plt.imshow(td_frame)

    # df = pd.DataFrame(np.array(event_data),
    #                   columns=["frame_now", "event_id", "prev_frame", "x", "y",
    #                            "polarity", "timestamp"])
    # df.to_csv(os.path.join(config.get('paths', 'log_path'),
    #                        "{}.csv".format('output')), index=False)
