from sacrebleu import BLEU
import re

blue = BLEU()


def calculate_bleu(predictions, references):
    # this function preprocess predictions before bleu calculation
    prediction = predictions[0].strip()
    if prediction:
        prediction = prediction.replace("\n", " ")
        # replace multi spaces with single space using re
        prediction = re.sub(r"\s+", " ", prediction)
    return prediction, references[0]


def calculate_blue_agg(items):
    predictions, references = zip(*items)
    return blue.corpus_score(predictions, [references]).score
