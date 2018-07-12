import os
from PyAedatTools.ImportAedat import ImportAedat


def configure_aedat_import(filename, config):
    aedat = {}
    aedat['importParams'] = {}
    aedat['importParams']['filePath'] = \
        os.path.join(config.get('paths', 'dataset_path'), filename)

    aedat['importParams']['startEvent'] = int(1e6)
    aedat['importParams']['endEvent'] = int(10e6)
    # aedat['importParams']['startTime'] = 48;
    # aedat['importParams']['endTime'] = 49;

    aedat['importParams']['dataTypes'] = {'polarity', 'special', 'frame'}

    return aedat


def load_aedat(aedat):
    aedat = ImportAedat(aedat)
    print('Read {} seconds of data'.format(
        (aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp'])
        / 1e6))
    print('Read {} events.'.format(
        aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp']))
    print('Done!')
