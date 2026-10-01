import torch
import string

def exact_match_score(predictions: list[str], references: list[str]) -> torch.Tensor:
    """
    Calculate the exact match score between predictions and references using PyTorch.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a torch.Tensor scalar (float32) between 0 and 1
    """
    # Your code here
    if not predictions: 
        return torch.tensor(0.0, dtype = torch.float32)

    table = str.maketrans('','',string.punctuation)
    def normalize(text):
        text = text.lower().translate(table)
        return ' '.join(text.split())

    matches = torch.tensor(
        [
        normalize(pred) == normalize(ref)
        for pred, ref in zip(predictions, references)
        ],
        dtype = torch.float32
    )

    return matches.mean()