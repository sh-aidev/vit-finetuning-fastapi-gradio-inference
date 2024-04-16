from typing import Dict, List, Tuple
from datasets import Dataset
import torch
import numpy as np
import evaluate
from transformers import ViTImageProcessor
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Resize, CenterCrop

from src.utils.logger import logger

metric = evaluate.load("accuracy")

def get_id_labels(dataset: Dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = dataset.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return label2id, id2label

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.debug(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def split_ds(dataset: Dataset, test_size: float, image_processor: ViTImageProcessor) -> Tuple[Dataset, Dataset]:
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )
    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch
    
    splits = dataset.train_test_split(test_size=test_size)
    train_ds = splits["train"]
    val_ds = splits["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    return train_ds, val_ds