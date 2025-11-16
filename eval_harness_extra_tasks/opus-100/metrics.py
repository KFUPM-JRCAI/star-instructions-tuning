from sacrebleu import BLEU

blue = BLEU()


def calculate_bleu(predictions, references):
    # this function preprocess predictions before bleu calculation
    prediction = predictions[0].strip()
    if prediction:
        prediction = prediction.splitlines()[0].strip()
    return prediction, references[0]


def calculate_blue_agg(items):
    predictions, references = zip(*items)
    return blue.corpus_score(predictions, [references]).score
