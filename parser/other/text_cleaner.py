import re

# Просто чистим текст
def simple_clean(text: str) -> str:
    text = text.replace('\\\\n', ' ').replace('\\\\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text