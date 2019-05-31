import itertools

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

def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def _recon_lcs(x, y):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = _lcs(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
    return recon_tuple

def _union_lcs(evaluated_sentences, reference_sentence, prev_union=None):
    """
    Returns LCS_u(r_i, C) which is the LCS score of the union longest common
    subsequence between reference sentence ri and candidate summary C.
    For example:
    if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8
    and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1
    is "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5".
    The union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5"
    and LCS_u(r_i, C) = 4/5.
    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries
    Returns:
      float: LCS_u(r_i, C)
    ValueError:
      Raises exception if a param has len <= 0
    """
    if prev_union is None:
        prev_union = set()

    if len(evaluated_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    lcs_union = prev_union
    prev_count = len(prev_union)

    combined_lcs_length = 0
    for eval_s in evaluated_sentences:
        lcs = set(_recon_lcs(reference_sentence, eval_s))
        combined_lcs_length += len(lcs)
        lcs_union = lcs_union.union(lcs)

    new_lcs_count = len(lcs_union) - prev_count
    return new_lcs_count, lcs_union

def _combine_sentences(sentences):
    return list(itertools.chain(*sentences))

def rouge_l(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (summary level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
    P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    SUM(i,u) = SUM from i through u
    u = number of sentences in reference summary
    C = Candidate summary made up of v sentences
    m = number of words in reference summary
    n = number of words in candidate summary
    Args:
      evaluated_sentences: The sentences that have been picked by the
                           summarizer
      reference_sentence: One of the sentences in the reference summaries
    Returns:
      A float: F_lcs
    Raises:
      ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    evaluate_sentences = [_remove_eos_pad(s) for s in evaluated_sentences]
    reference_sentences = [_remove_eos_pad(s) for s in reference_sentences]

    # total number of words in reference sentences
    m = len(set(_combine_sentences(reference_sentences)))

    # total number of words in evaluated sentences
    n = len(set(_combine_sentences(evaluated_sentences)))

    union_lcs_sum_across_all_references = 0
    union = set()
    for ref_s in reference_sentences:
        lcs_count, union = _union_lcs(evaluated_sentences,
                                      ref_s,
                                      prev_union=union)
        union_lcs_sum_across_all_references += lcs_count

    llcs = union_lcs_sum_across_all_references
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs 
    # return {"f": f_lcs, "p": p_lcs, "r": r_lcs}
