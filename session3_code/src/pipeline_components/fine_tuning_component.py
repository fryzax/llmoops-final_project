"""Fine-tuning component for Vertex AI pipeline."""

from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0",
        "transformers==4.46.*",
        "peft==0.13.2",
        "accelerate==1.10.1",
        "trl==0.17.0",
        "bitsandbytes==0.47.0",
        "datasets==4.0.0",
        "huggingface-hub==0.34.4",
        "safetensors==0.6.2",
        "pandas==2.2.2",
        "numpy==2.0.2",
        "tensorboard",
        "gcsfs",
    ],
)
def fine_tuning_component(
    dataset: Input[Dataset], metrics: Output[Metrics], model: Output[Model]
):
    """Fine-tune a Phi-3 model using LoRA and integrate with Vertex AI."""
    import logging
    import time

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig,  # pyright: ignore[reportPrivateImportUsage]
        get_peft_model,  # pyright: ignore[reportPrivateImportUsage]
        prepare_model_for_kbit_training,  # pyright: ignore[reportPrivateImportUsage]
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting fine tuning process...")

    hyperparameters = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "val_split_ratio": 0.2,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 3e-4,
        "num_epochs": 10,
        "batch_size": 16,
        "max_length": 64,
        "gradient_accumulation_steps": 1,
    }

    logger.info("Creating training configurations...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    lora_config = LoraConfig(
        r=hyperparameters["lora_r"],
        lora_alpha=hyperparameters["lora_alpha"],
        bias="none",
        lora_dropout=hyperparameters["lora_dropout"],
        task_type="CAUSAL_LM",
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
    )
    sft_config = SFTConfig(
        output_dir=model.path,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        per_device_train_batch_size=hyperparameters["batch_size"],
        auto_find_batch_size=True,
        max_length=hyperparameters["max_length"],
        packing=True,
        num_train_epochs=hyperparameters["num_epochs"],
        learning_rate=hyperparameters["learning_rate"],
        optim="paged_adamw_8bit",
        logging_steps=1,
        logging_dir=metrics.path,
        report_to="tensorboard",
        bf16=torch.cuda.is_bf16_supported(including_emulation=False),
        do_eval=True,
        eval_strategy="epoch",
    )

    logger.info("Loading pre-trained model...")
    pre_trained_model = AutoModelForCausalLM.from_pretrained(
        hyperparameters["model_name"],
        device_map="cuda:0",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    pre_trained_model = prepare_model_for_kbit_training(pre_trained_model)
    pre_trained_model = get_peft_model(pre_trained_model, lora_config)

    logger.info(f"Loading dataset from {dataset.path}...")
    full_dataset = Dataset.from_pandas(
        pd.read_csv(dataset.path).assign(
            messages=lambda df: df["messages"].apply(
                lambda x: eval(x.replace("\n", ","))
            )
        )
    ).train_test_split(test_size=hyperparameters["val_split_ratio"])
    train_dataset, eval_dataset = (
        full_dataset["train"],
        full_dataset["test"],
    )

    logger.info("Creating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["model_name"])
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    logger.info("Starting training...")
    trainer = SFTTrainer(
        model=pre_trained_model.base_model.model,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    training_start_time = time.time()
    trainer.train()
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time

    logger.info(f"Saving model at {model.path}...")
    trainer.save_model(model.path)

    logger.info("Logging metrics...")
    metrics.log_metric("training_time", training_duration)
    for metric_name, metric_value in hyperparameters.items():
        metrics.log_metric(metric_name, metric_value)
