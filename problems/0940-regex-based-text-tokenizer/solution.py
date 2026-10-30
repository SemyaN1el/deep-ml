import re

def tokenize_text(text: str) -> list:
    """
    Split raw text into tokens using regex-based splitting on whitespace
    and punctuation. Returns a list of non-empty stripped tokens.
    """
    pattern = r"""(--|[,.:;?_!"()']|\s+)"""

    tokens = re.split(pattern, text)

    return [token.strip() for token in tokens if token.strip()]