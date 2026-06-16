import evaluate
import sys

from pathlib import Path
from dotenv import load_dotenv
from radfact_lite import ModelConfig, PipelineModels, ReportType, compute_radfact_text

predictions = ["a dog running trough the grass"]
references = ["in the grass field  there is a bau sprinting"]

candidates_by_id = {str(index): prediction for index, prediction in enumerate(predictions)}
references_by_id = {str(index): reference for index, reference in enumerate(references)}


bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

bleu_result = bleu.compute(
    predictions=predictions,
    references=references,
    max_order=4,
)


meteor_result = meteor.compute(
    predictions=predictions,
    references=references,
)

print(f"BLEU-4: {bleu_result}")
print(f"METEOR: {meteor_result}")


try:
    radfact_model = ModelConfig(model="gpt-4o-mini")
    radfact_models = PipelineModels(
        parse_model=radfact_model,
        entailment_model=radfact_model,
        filtering_model=radfact_model,
    )
    radfact_result = compute_radfact_text(
        candidates_by_id=candidates_by_id,
        references_by_id=references_by_id,
        models=radfact_models,
        report_type=ReportType.TOOTHFAIRY,
        is_narrative_text=True,
        filter_negatives=False,
    )
    print(f"RadFact-Lite: {radfact_result}")
except Exception as error:
    print(f"RadFact-Lite could not be computed: {error}")
