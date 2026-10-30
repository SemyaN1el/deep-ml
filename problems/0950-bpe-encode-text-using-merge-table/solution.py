def bpe_encode(text, token_to_id, merges):
    tokens = [token_to_id[ch] for ch in text]

    while len(tokens) > 1:
        bestPair = None 
        bestId = None
        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])

            if pair in merges:
                newId = merges[pair]

                if bestId is None or newId < bestId:
                    bestPair = pair 
                    bestId = newId 
            
        if bestPair is None:
            break 
        ans = []
        i = 0
        while i < len(tokens):
            pair = (tokens[i], tokens[i+1] if i + 1 < len(tokens) else None)

            if pair == bestPair: 
                ans.append(bestId)
                i += 2
            else:
                ans.append(tokens[i])
                i += 1
        
        tokens = ans 

    
    return tokens 
