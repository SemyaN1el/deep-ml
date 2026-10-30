def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    ans = []

    for token_id in ids:
        token = vocab[token_id]

        if token.startswith("G"):
            token = ' ' + token[1:]
        
        ans.append(token)
    
    return ''.join(ans)
        
