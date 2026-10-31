def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    words = {
        tuple(word.split()): freq for word, freq in corpus.items()
    }
    mergeHistory = []

    for _ in range(num_merges):
        pairCounts = {}

        for tokens, freq in words.items():
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                pairCounts[pair] = pairCounts.get(pair,0) + freq
        
        if not pairCounts:
            break

        bestPair = max(pairCounts, key = pairCounts.get)
        mergeHistory.append(bestPair)

        newWords = {}

        for tokens, freq in words.items():
            ans = []
            i = 0

            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == bestPair:
                    ans.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    ans.append(tokens[i])
                    i += 1
            
            mergedTokens = tuple(ans)
            newWords[mergedTokens] = newWords.get(mergedTokens, 0) + freq
        words = newWords
    
    return mergeHistory