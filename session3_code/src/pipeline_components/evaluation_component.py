"""Evaluation component using RAGAS library."""

from kfp.dsl import Dataset, Input, Metrics, Output, OutputPath, component


@component(
    base_image="cicirello/pyaction:3.11",
    packages_to_install=[
        "ragas>=0.3.5",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.5.1",
        "pandas>=2.3.2",
        "tqdm",
    ],
)
def evaluation_component(
    predictions: Input[Dataset],
    metrics: Output[Metrics],
    evaluation_results: OutputPath("Dataset"),  # type: ignore
):
    """Computes evaluation metrics on test set predictions."""
    import logging

    import pandas as pd
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, RougeScore
    from ragas.metrics.base import SingleTurnMetric
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def compute_metrics(
        user_input: str,
        response: str,
        reference: str,
        metric_definitions: list[SingleTurnMetric],
    ) -> dict[str, float | int]:
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference,
        )
        return {
            metric_definition.name: metric_definition.single_turn_score(sample)
            for metric_definition in metric_definitions
        }

    def compute_aggregated_metrics(
        evaluations_df: pd.DataFrame, metric_definitions: list[SingleTurnMetric]
    ) -> dict[str, float]:
        return {
            f"avg_{col}": evaluations_df[col].mean()
            for col in [
                metric_definition.name for metric_definition in metric_definitions
            ]
        }

    logger.info(f"Loading predictions from {predictions.path}")
    predictions_df = pd.read_csv(predictions.path)

    metric_definitions = [BleuScore(), RougeScore()]

    logger.info("Computing evaluation metrics...")
    evaluations = []
    for _, row in tqdm(predictions_df.iterrows(), total=predictions_df.shape[0]):
        evaluations.append(
            compute_metrics(
                row["user_input"],
                row["response"],
                row["reference"],
                metric_definitions,
            )
        )

    evaluations_df = pd.concat([predictions_df, pd.DataFrame(evaluations)], axis=1)

    logger.info(f"Writing evaluation results to {evaluation_results}...")
    evaluations_df.to_csv(evaluation_results, index=False)

    for metric_name, metric_value in compute_aggregated_metrics(
        evaluations_df, metric_definitions
    ).items():
        metrics.log_metric(metric_name, metric_value)
