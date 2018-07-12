# -*- coding: utf-8 -*-
"""
This module bundles all the steps in the spikeGANs pipeline.

Important functions:

.. autosummary::
    :nosignatures:

    run_full
    update_setup

@author: rbodo
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os
from future import standard_library

standard_library.install_aliases()


def run_full(config):
    """Test various frame generation algorithms on different datasets.

    Pipeline:
        1.

    Parameters
    ----------

    config: configparser.ConfigParser
        ConfigParser containing the user settings.

    """

    from spikeGANs.datasets.aedat import configure_aedat_import, load_aedat
    from spikeGANs.frame_generation.event_surface import get_frames

    aedat = configure_aedat_import('Tobi2.aedat', config)

    load_aedat(aedat)

    get_frames(aedat, config)


def load_config(filepath):
    """ Load a config file from ``filepath``. """

    # Import takes into account both python 2 and 3.
    try:
        import configparser
    except ImportError:
        # noinspection PyPep8Naming, PyUnresolvedReferences
        import ConfigParser as configparser
        configparser = configparser

    assert os.path.isfile(filepath), \
        "Configuration file not found at {}.".format(filepath)

    config = configparser.ConfigParser()
    config.read(filepath)

    return config


def configure_paths(config):
    """Configure the output folders."""

    log_path = config.get('paths', 'log_path')
    generator_image_folder = os.path.join(log_path, 'generator')
    target_image_folder = os.path.join(log_path, 'targets')
    combined_image_folder = os.path.join(log_path, 'combined')
    if not os.path.exists(generator_image_folder):
        os.makedirs(generator_image_folder)
    if not os.path.exists(target_image_folder):
        os.makedirs(target_image_folder)
    if not os.path.exists(combined_image_folder):
        os.makedirs(combined_image_folder)

    config.set('paths', 'generator_image_path', generator_image_folder)
    config.set('paths', 'target_image_path', target_image_folder)
    config.set('paths', 'combined_image_path', combined_image_folder)

    return config


def update_setup(config_filepath):
    """Update default settings with user settings and check they are valid.

    Load settings from configuration file at ``config_filepath``, and check
    that parameter choices are valid. Non-specified settings are filled in with
    defaults.
    """

    # Load defaults.
    config = load_config(os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'config_defaults')))

    # Overwrite with user settings.
    config.read(config_filepath)

    config = configure_paths(config)

    # Set default log path if user did not specify it.
    log_path = config.get('paths', 'log_path')
    if log_path == '':
        log_path = os.path.join(os.path.dirname(config_filepath), 'log')
        config.set('paths', 'log_path', log_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    plot_var = get_plot_keys(config)
    plot_vars = config_string_to_set_of_strings(config.get('restrictions',
                                                           'plot_vars'))
    assert all([v in plot_vars for v in plot_var]), \
        "Plot variable(s) {} not understood.".format(
            [v for v in plot_var if v not in plot_vars])
    if 'all' in plot_var:
        plot_vars_all = plot_vars.copy()
        plot_vars_all.remove('all')
        config.set('output', 'plot_vars', str(plot_vars_all))

    log_var = get_log_keys(config)
    log_vars = config_string_to_set_of_strings(config.get('restrictions',
                                                          'log_vars'))
    assert all([v in log_vars for v in log_var]), \
        "Log variable(s) {} not understood.".format(
            [v for v in log_var if v not in log_vars])
    if 'all' in log_var:
        log_vars_all = log_vars.copy()
        log_vars_all.remove('all')
        config.set('output', 'log_vars', str(log_vars_all))

    # Change matplotlib plot properties, e.g. label font size
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
        if len(plot_vars) > 0:
            import warnings
            warnings.warn("Package 'matplotlib' not installed; disabling "
                          "plotting. Run 'pip install matplotlib' to enable "
                          "plotting.", ImportWarning)
            config.set('output', 'plot_vars', str({}))
    if matplotlib is not None:
        matplotlib.rcParams.update(eval(config.get('output',
                                                   'plotproperties')))

    with open(os.path.join(log_path, '.config'), str('w')) as f:
        config.write(f)

    return config


def get_log_keys(config):
    return config_string_to_set_of_strings(config.get('output', 'log_vars'))


def get_plot_keys(config):
    return config_string_to_set_of_strings(config.get('output', 'plot_vars'))


def config_string_to_set_of_strings(string):
    set_unicode = set(eval(string))
    return {str(s) for s in set_unicode}
