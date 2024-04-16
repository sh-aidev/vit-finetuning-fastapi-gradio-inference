import os
from src.utils.logger import logger
from src.utils.config import Config
from src.core.training import FinetuningTraining
from src.core.inference import VITInference, VITInferenceHF
# from src.server.server import VITServer

class App:
    """
    Main application class to run the FastAPI server. This class will initialize the server and run it.
    """
    def __init__(self) -> None:
        root_config_dir = "configs"
        logger.debug(f"Root config dir: {root_config_dir}")
        self.config = Config(root_config_dir)
        if self.config.vit_config.task_name == "train":
            logger.info("Finetuning mode")
            self.vit = FinetuningTraining(self.config)
        elif self.config.vit_config.task_name == "infer":
            self.vit = VITInferenceHF(self.config)
        # elif self.config.vit_config.task_name == "server":
        #     self.vit = LLMServer(self.config)
    
    def run(self):
        if self.config.vit_config.task_name == "infer":
            if self.config.vit_config.hf.push_huggingface == True:
                self.vit.push_to_huggingface()
            image_url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
            self.vit.run(image_url)
        else:
            self.vit.run()