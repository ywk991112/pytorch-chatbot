def _get_word_ngrams(n, sentence):
    assert len(sentence.size()) == 1
    sentence = sentence.tolist()
    tmp_list = [sentence[i:] for i in range(n)]
    return set(zip(*tmp_list))

def _remove_eos_pad(sentence):
    EOS_position = (sentence == 1).nonzero()
    if not EOS_position.size(0) == 0:
        return sentence[:EOS_position[0].item()]
    return sentence

def rouge_n(evaluated_sentence, reference_sentence, n):
    """
    Computes ROUGE-N of two text collections of sentences.
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentences: The sentences from the referene set
      n: Size of ngram.  Defaults to 2.
    Returns:
      A tuple (f1, precision, recall) for ROUGE-N
    """
    if len(evaluated_sentence) <= 0 or len(reference_sentence) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluated_sentence = _remove_eos_pad(evaluated_sentence)
    reference_sentence = _remove_eos_pad(reference_sentence)

    evaluated_ngram = _get_word_ngrams(n, evaluated_sentence)
    reference_ngram = _get_word_ngrams(n, reference_sentence)
    reference_count = len(reference_ngram)
    evaluated_count = len(evaluated_ngram)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngram = evaluated_ngram.intersection(reference_ngram)
    overlapping_count = len(overlapping_ngram)

    # Handle edge case. This isn't mathematically correct, but it's good enough
    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

    return f1_score 
