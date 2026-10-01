import string

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    # Your code here
    if not predictions: 
        return 0 

    table = str.maketrans('','',string.punctuation)
    def normalize(text):
        text = text.lower().translate(table)
        return ' '.join(text.split())

    matches = sum(normalize(pred) == normalize(ref) for pred, ref in zip(predictions, references))
    return matches / len(predictions)