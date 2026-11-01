import numpy as np

def bigram_counts(words):
    # Your code here
    chars = sorted(set(''.join(words)) - {'.'})
    vocab = ['.'] + chars 

    stoi = {ch: i for i, ch in enumerate(vocab)}

    V = len(vocab)
    N = np.zeros((V,V), dtype=int)

    for word in words:
        seq = ['.'] + list(word) + ['.']

        for a, b in zip(seq, seq[1:]):
            N[stoi[a], stoi[b]] += 1

    return N.tolist()