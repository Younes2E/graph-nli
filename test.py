import spacy
nlp = spacy.load('en_core_web_sm')

def find_token_span(doc, entity):
    char_start = doc.text.lower().find(entity.lower())
    if char_start == -1:
        return None
    char_end = char_start + len(entity) - 1
    start_tok = end_tok = None
    for i, tok in enumerate(doc):
        if start_tok is None and tok.idx <= char_start < tok.idx + len(tok.text):
            start_tok = i
        if tok.idx <= char_end < tok.idx + len(tok.text):
            end_tok = i
    if start_tok is not None and end_tok is not None:
        return (start_tok, end_tok)
    return None



    return "\n".join(triplets)