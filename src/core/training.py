from src.utils.config import Config
from src.utils.logger import logger
from src.utils.utils import get_id_labels, split_ds, print_trainable_parameters, compute_metrics, collate_fn

from pathlib import Path
import os
from transformers import ViTImageProcessor, AutoModelForImageClassification,  TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


class FinetuningTraining():
    def __init__(self, config: Config) -> None:
        logger.debug(f"Initializing FinetuningTraining...")
        dataset = load_dataset(config.vit_config.data.path, split = config.vit_config.data.split)
        logger.debug(f"Dataset loaded...")

        label2id, id2label = get_id_labels(dataset)

        image_processor = ViTImageProcessor.from_pretrained(config.vit_config.vit_model.pretrained_model_name_or_path)
        logger.debug(f"Image processor loaded...")

        train_ds, val_ds = split_ds(dataset, config.vit_config.data.test_size, image_processor)

        logger.debug(f"Train and validation datasets split completed...")
        model = AutoModelForImageClassification.from_pretrained(
            config.vit_config.vit_model.pretrained_model_name_or_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=config.vit_config.vit_model.ignore_mismatched_sizes,
        )

        print_trainable_parameters(model)

        logger.debug(f"Model loaded...")

        lora_config = LoraConfig(
            r=config.vit_config.lora.r,
            lora_alpha=config.vit_config.lora.lora_alpha,
            target_modules=config.vit_config.lora.target_modules,
            lora_dropout=config.vit_config.lora.lora_dropout,
            bias=config.vit_config.lora.bias,
            modules_to_save=config.vit_config.lora.modules_to_save,
        )
        lora_model = get_peft_model(model, lora_config)
        logger.debug(f"PEFT model loaded...")

        print_trainable_parameters(lora_model)
        self.output_dir = Path(config.vit_config.paths.output_dir)
        self.log_dir = Path(config.vit_config.paths.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(self.output_dir, "checkpoints")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output and log directories created...")
        model_name = config.vit_config.vit_model.pretrained_model_name_or_path.split("/")[-1]
        batch_size = 128

        args = TrainingArguments(
            output_dir=os.path.join(self.output_dir,f"{model_name}-finetuned-lora-food101"),
            logging_dir=self.log_dir,
            remove_unused_columns=config.vit_config.vit_trainer.remove_unused_columns,
            per_device_train_batch_size=config.vit_config.vit_trainer.batch_size,
            gradient_accumulation_steps=config.vit_config.vit_trainer.gradient_accumulation_steps,
            per_device_eval_batch_size=config.vit_config.vit_trainer.batch_size,
            learning_rate=config.vit_config.vit_trainer.learning_rate,
            logging_steps=config.vit_config.vit_trainer.logging_steps,
            fp16=config.vit_config.vit_trainer.fp16,
            save_strategy=config.vit_config.vit_trainer.save_strategy,
            evaluation_strategy=config.vit_config.vit_trainer.evaluation_strategy,
            num_train_epochs=config.vit_config.vit_trainer.num_train_epochs,
            load_best_model_at_end=config.vit_config.vit_trainer.load_best_model_at_end,
            metric_for_best_model=config.vit_config.vit_trainer.metric_for_best_model,
            label_names=config.vit_config.vit_trainer.label_names,
            report_to=config.vit_config.vit_trainer.report_to,
        )

        logger.debug(f"Training arguments loaded...")

        self.trainer = Trainer(
            model = lora_model,
            args = args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        logger.debug(f"Trainer loaded...")
    
    def run(self):
        logger.debug(f"Training started...")
        train_results = self.trainer.train()
        logger.debug(f"Training completed...")
        logger.debug(f"Saving model...")
        self.trainer.save_model(self.model_dir)
        logger.debug(f"Model saved...")