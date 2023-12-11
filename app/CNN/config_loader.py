# This class stays tightly coupled with the config file/files. This allows loose coupling between config and 
# souurce code. This logic could easily be altered to read any other config files, without touching the main
# source.
import yaml

class Loader():
    @staticmethod
    def get_config():
        with open("config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        return config