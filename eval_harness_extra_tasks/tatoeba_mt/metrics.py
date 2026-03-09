from sacrebleu import BLEU
import logging

logger = logging.getLogger(__name__)

bleu = BLEU()

_empty_prediction_count = 0


def calculate_bleu(predictions, references):
    global _empty_prediction_count
    # this function preprocess predictions before bleu calculation
    prediction = predictions[0].strip()
    if prediction:
        prediction = prediction.splitlines()[0].strip()
    else:
        _empty_prediction_count += 1
        logger.warning(f"Empty prediction encountered (total so far: {_empty_prediction_count})")
    return prediction, references[0]


def calculate_bleu_agg(items):
    global _empty_prediction_count
    predictions, references = zip(*items)
    if _empty_prediction_count > 0:
        logger.warning(f"Total empty predictions: {_empty_prediction_count}/{len(items)}")
        _empty_prediction_count = 0
    return bleu.corpus_score(predictions, [references]).score
