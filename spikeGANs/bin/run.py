# -*- coding: utf-8 -*-

"""
The purpose of this module is to provide an executable for running the
spikeGANs software from the terminal.

@author: rbodo
"""

import argparse
import os


def main(filepath=None):
    """Entry point for running the toolbox.
    """

    from spikeGANs.bin.utils import update_setup, run_full

    parser = argparse.ArgumentParser(
        description='Encode events from DVS sensor in feature space using '
                    'GANs.')
    parser.add_argument('config_filepath', help='Path to configuration file.',
                        nargs='?')
    args = parser.parse_args()

    if filepath is None:
        filepath = os.path.abspath(args.config_filepath)

    assert os.path.isfile(filepath), \
        "Configuration file not found at {}.".format(filepath)

    config = update_setup(filepath)

    run_full(config)


if __name__ == '__main__':
    fp = None
    # fp = 'C:\\Users\\bodor\\PycharmProjects\\spikeGANs\\examples\\config'
    main(fp)
