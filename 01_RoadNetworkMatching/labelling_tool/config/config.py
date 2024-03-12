import configparser
import os


class ConfigException(Exception):
    pass


def read_config():
    config = configparser.ConfigParser()

    # read correct config file
    if os.path.isfile('config/config.ini'):
        config.read('config/config.ini')
    elif os.path.isfile('config/config_default.ini'):
        config.read('config/config_default.ini')
    elif os.path.isfile('labelling_tool/config/config.ini'):
        config.read('labelling_tool/config/config.ini')
    elif os.path.isfile('labelling_tool/config/config_default.ini'):
        config.read('labelling_tool/config/config_default.ini')
    else:
        raise ConfigException("There is no config.ini or config_default.ini in the config folder. Create a config file and try again.")

    try:
        data = config['DATA']
    except KeyError:
        raise ConfigException("Malformatted config file. DATA section is missing.")
    try:
        path_osm_network = data['pathosmnetwork']
        path_tomtom_network = data['pathtomtomnetwork']
        path_sumolib = data['pathsumolib']
    except KeyError:
        raise ConfigException("Malformatted config file. There must be a 'pathosmnetwork' and 'pathtomtomnetwork' attribute in the 'DATA' section.")
    
    return path_osm_network, path_tomtom_network, path_sumolib
    



