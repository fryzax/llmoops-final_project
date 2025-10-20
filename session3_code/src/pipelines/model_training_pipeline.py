"""Model training pipeline definition for Vertex AI."""

from kfp.dsl import pipeline

from src.pipeline_components.data_transformation_component import (
    data_transformation_component,
)
from src.pipeline_components.evaluation_component import evaluation_component
from src.pipeline_components.fine_tuning_component import fine_tuning_component
from src.pipeline_components.inference_component import inference_component


@pipeline(name="model-training-pipeline")
def model_training_pipeline(
    raw_dataset_uri: str,
) -> None:
    """Model training pipeline definition."""
    data_transformation_task = data_transformation_component(
        train_test_split_ratio=0.1,
        raw_dataset_uri=raw_dataset_uri,
    )  # type: ignore

    fine_tuning_task = fine_tuning_component(
        dataset=data_transformation_task.outputs["train_dataset"]
    )  # type: ignore

    (
        fine_tuning_task.set_accelerator_type("NVIDIA_TESLA_T4")
        .set_cpu_limit("16")
        .set_memory_limit("50G")
    )

    inference_task = inference_component(  # type: ignore
        dataset=data_transformation_task.outputs["test_dataset"],
        model=fine_tuning_task.outputs["model"],
    )

    (
        inference_task.set_accelerator_type("NVIDIA_TESLA_T4")
        .set_cpu_limit("16")
        .set_memory_limit("50G")
    )

    evaluation_component(  # type: ignore
        predictions=inference_task.outputs["predictions"],
    )
