"""Evaluation component using RAGAS library."""

from kfp.dsl import Dataset, Input, Metrics, Output, OutputPath, component


@component(
    base_image="cicirello/pyaction:3.11",
    packages_to_install=[
        "ragas>=0.3.5",
        "rouge-score>=0.1.2",
            "sacrebleu>=2.5.1",
            # for BERTScore
            "evaluate>=0.4.0",
            "bert-score",
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
    # faithfulness metric lives in ragas.metrics.faithfulness
    # RAGAS metric classes follow the <Name>Score naming convention
    try:
        from ragas.metrics.faithfulness import FaithfulnessScore
    except Exception:  # pragma: no cover - defensive import in case of older/newer ragas
        # Fallback: try to import the module safely and extract a likely attribute name
        try:
            import ragas.metrics.faithfulness as _faith_mod

            FaithfulnessScore = getattr(_faith_mod, "FaithfulnessScore", None)
            if FaithfulnessScore is None:
                # Last resort: try 'Faithfulness' name
                FaithfulnessScore = getattr(_faith_mod, "Faithfulness", None)
        except Exception:
            # Module not available or import error; leave as None and continue
            FaithfulnessScore = None
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
    # Add faithfulness metric if available
    if "FaithfulnessScore" in locals() and FaithfulnessScore is not None:
        metric_definitions.append(FaithfulnessScore())
    else:
        logger.warning(
            "Faithfulness metric not available from ragas.metrics.faithfulness; skipping.")

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

    # Compute BERTScore (semantic similarity) over the whole dataset if available.
    # We compute per-sample F1 and add it as a column 'BERTScore'.
    try:
        from evaluate import load as _load_eval

        logger.info("Computing BERTScore (semantic similarity) over dataset...")
        _bertscore = _load_eval("bertscore")
        preds = evaluations_df["response"].fillna("").astype(str).tolist()
        refs = evaluations_df["reference"].fillna("").astype(str).tolist()
        # language: set to 'fr' as your examples are French; adjust if needed
        _bs_res = _bertscore.compute(predictions=preds, references=refs, lang="fr")
        # _bs_res typically contains lists for 'precision','recall','f1'
        f1_list = _bs_res.get("f1")
        if f1_list and len(f1_list) == len(evaluations_df):
            evaluations_df["BERTScore"] = f1_list
        else:
            # If backend returned a scalar or unexpected shape, try to coerce
            try:
                evaluations_df["BERTScore"] = list(_bs_res.get("f1", []))
            except Exception:
                logger.warning("Unexpected BERTScore result shape; skipping per-sample values.")
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning(f"BERTScore computation skipped: {e}")

    logger.info(f"Writing evaluation results to {evaluation_results}...")
    evaluations_df.to_csv(evaluation_results, index=False)

    for metric_name, metric_value in compute_aggregated_metrics(
        evaluations_df, metric_definitions
    ).items():
        metrics.log_metric(metric_name, metric_value)

    # Log average BERTScore if present
    if "BERTScore" in evaluations_df.columns:
        try:
            metrics.log_metric("avg_BERTScore", evaluations_df["BERTScore"].mean())
        except Exception:
            logger.warning("Could not log avg_BERTScore metric")
