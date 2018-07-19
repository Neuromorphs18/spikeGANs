import os
from PyAedatTools.ImportAedat import ImportAedat


def load_aedat(config):
    aedat = {}
    aedat['importParams'] = {}
    aedat['importParams']['filePath'] = \
        os.path.join(config.get('paths', 'dataset_path'),
                     config.get('paths', 'filename'))
    if config.get('input', 'start_event') != '':
        aedat['importParams']['startEvent'] = \
            int(config.getfloat('input', 'start_event'))
    if config.get('input', 'end_event') != '':
        aedat['importParams']['endEvent'] = \
            int(config.getfloat('input', 'end_event'))

    aedat = ImportAedat(aedat)
    print("Read {} events during {} seconds stream.".format(
        len(aedat['data']['polarity']['timeStamp']),
        (aedat['info']['lastTimeStamp'] - aedat['info']['firstTimeStamp'])
        / 1e6))

    return aedat
