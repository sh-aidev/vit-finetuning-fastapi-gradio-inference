from peft import PeftConfig, PeftModel
from pathlib import Path
from transformers import AutoModelForImageClassification, AutoImageProcessor, ViTImageProcessor
from datasets import load_dataset
from src.utils.utils import get_id_labels
from PIL import Image
from typing import Union, Dict
import requests, os
import torch

from src.utils.config import Config
from src.utils.logger import logger

class VITInference():
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        logger.debug(f"Initializing VITInference...")
        dataset = load_dataset(cfg.vit_config.data.path, split = cfg.vit_config.data.split)
        logger.debug(f"Dataset loaded...")

        label2id, id2label = get_id_labels(dataset)
        output_dir = cfg.vit_config.paths.output_dir
        model_dir = os.path.join(output_dir, "checkpoints")
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)

        peft_config = PeftConfig.from_pretrained(model_dir)
        logger.debug(f"Peft Config loaded...")
        model = AutoModelForImageClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=cfg.vit_config.vit_model.ignore_mismatched_sizes,
        )
        logger.debug(f"Model loaded...")
        print(model_dir)
        self.inference_model = PeftModel.from_pretrained(model, model_dir)
        logger.debug(f"PEFT model loaded...")

    def run(self, img: Union[str, Image.Image]) -> Dict[str, str]:
        logger.debug(f"Running inference...")
        
        if isinstance(img, Image.Image):
            image = Image.open(img)
        elif isinstance(img, str):
            image = Image.open(requests.get(img, stream=True).raw)
        logger.debug(f"Image loaded...")
        
        encoding = self.image_processor(image.convert("RGB"), return_tensors="pt")
        logger.debug(f"Encoding done...")

        with torch.no_grad():
            outputs = self.inference_model(**encoding)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        logger.info(f"Predicted class: {self.inference_model.config.id2label[predicted_class_idx]}")

        return {"class": self.inference_model.config.id2label[predicted_class_idx]}

    def push_to_huggingface(self) -> None:
        logger.debug(f"Pushing model to Huggingface...")
        merged_model = self.inference_model.merge_and_unload()
        logger.debug(f"Model merged and unloaded...")
        merged_model.push_to_hub(self.cfg.vit_config.hf.hf_model_id)
        logger.debug(f"Model pushed to Huggingface...")
        self.image_processor.push_to_hub(self.cfg.vit_config.hf.hf_model_id)
        logger.debug(f"Tokenizer pushed to Huggingface...")

class VITInferenceHF():
    def __init__(self, cfg: Config) -> None:
        logger.debug(f"Initializing VITInference for Huggingface...")
        dataset = load_dataset(cfg.vit_config.data.path, split = cfg.vit_config.data.split)
        logger.debug(f"Dataset loaded...")

        label2id, id2label = get_id_labels(dataset)
        logger.debug(f"Labels loaded...")

        self.image_processor = ViTImageProcessor.from_pretrained(cfg.vit_config.hf.hf_model_id)
        logger.debug(f"Image processor loaded...")

        self.model = AutoModelForImageClassification.from_pretrained(
            cfg.vit_config.hf.hf_model_id,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=cfg.vit_config.vit_model.ignore_mismatched_sizes,
        )
        logger.debug(f"Model loaded...")

    def run(self, img: Union[str, Image.Image]) -> Dict[str, str]:
        logger.debug(f"Running inference...")
        
        if isinstance(img, Image.Image):
            image = Image.open(img)
        elif isinstance(img, str):
            image = Image.open(requests.get(img, stream=True).raw)
        logger.debug(f"Image loaded...")
        
        encoding = self.image_processor(image.convert("RGB"), return_tensors="pt")
        logger.debug(f"Encoding done...")

        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            prediction = torch.nn.functional.softmax(logits, dim=-1)
        
        predicted_class_idx = prediction.argmax(-1).item()
        # get top k class index
        top_k = torch.topk(prediction, k=10)

        confidences = {self.model.config.id2label[i]: float(top_k.values[0].cpu().numpy()[j]) for j, i in enumerate(top_k.indices[0].cpu().numpy())}

        logger.info(f"Top Predicted class: {self.model.config.id2label[predicted_class_idx]}")

        return confidences
