import toml
import os
from src.utils.models import Model

class Config:

    def __init__(self, root_config_path: str):

        self.vit_config = Model(
            **toml.load(os.path.join(root_config_path, "config.toml"))
        )