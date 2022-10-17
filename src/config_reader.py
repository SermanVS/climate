import configparser

class Config:
    def __init__(self):
        config_parser = configparser.ConfigParser()
        config_parser.read('config.cfg')
        self.mode = config_parser['DEFAULT']['mode']
        self.metric = config_parser['DEFAULT']['metric']