from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.nn.functional as F
import torch

# See this for input references - https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu
# A Caption should be a list of strings.
# Reference Captions are list of actual captions - list(list(str))
# Predicted Caption is the string caption based on your model's output - list(str)
# Make sure to process your captions before evaluating bleu scores -
# Converting to lower case, Removing tokens like <start>, <end>, padding etc.

def bleu1(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)

def bleu4(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)

def get_caption(output, vocab, config):
    max_length = config['max_length']
    pred = output.data[:max_length]
    return [vocab.idx2word[idx.item()] for idx in pred]

def remove(captions):
    special_token = {'\pad', '\\bos', '\eos', '\\eos'}
    return [x for x in captions if x not in special_token]
