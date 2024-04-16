from pydantic import BaseModel

class Logger(BaseModel):
    environment: str

class Server(BaseModel):
    host: str
    port: int

class VITModel(BaseModel):
    pretrained_model_name_or_path: str
    ignore_mismatched_sizes: bool

class VITData(BaseModel):
    path: str
    split: str
    test_size: float

class PathConfig(BaseModel):
    output_dir: str
    log_dir: str

class VITTrainer(BaseModel):
    remove_unused_columns: bool
    gradient_accumulation_steps: int
    batch_size: int
    learning_rate: float
    logging_steps: int
    fp16: bool
    save_strategy: str
    evaluation_strategy: str
    num_train_epochs: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    label_names: list
    report_to: str

class Huggingface(BaseModel):
    push_huggingface: bool
    hf_model_id: str

class Lora(BaseModel):
    r: int
    lora_alpha: int
    target_modules: list
    lora_dropout: float
    bias: str
    modules_to_save: list

class Model(BaseModel):
    task_name: str
    server_type: str
    logger: Logger
    server: Server
    vit_model: VITModel
    data: VITData
    paths: PathConfig
    lora: Lora
    vit_trainer: VITTrainer
    hf: Huggingface