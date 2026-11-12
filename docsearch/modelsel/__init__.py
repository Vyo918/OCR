"""Task-specific OCR model selection (see MODEL_SELECTION.md).

Public leaderboards (OmniDocBench, OCRBench) measure *document parsing* and *scene-text QA*. Our task is
*text spotting on scanned CAD plots* (sparse, tiny, mixed Traditional-Chinese / numeric labels where one wrong
digit flips an answer, and the position of each label matters). So the candidates are scored on a ground
truth read by hand from the real drawings:  python -m docsearch.modelsel run|score
"""
