import configparser

class Config:
    def __init__(self):
        config_parser = configparser.ConfigParser()
        config_parser.read('config.cfg')
        self.mode = config_parser['DEFAULT']['mode']
        self.metric = config_parser['DEFAULT']['metric']
        self.metric_path = config_parser['DEFAULT']['metric_path']
        self.w = int(config_parser['DEFAULT']['w'])