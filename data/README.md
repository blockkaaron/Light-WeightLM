# Data

Large data files are excluded from git (see `.gitignore`).

## Directory Layout

```
data/
├── raw/          ← Place your .txt corpus files here (gitignored)
├── tokenized/    ← Output of tokenize_corpus.py — .bin files (gitignored)
└── README.md     ← This file — document your data sources here
```

## Corpus Sources

Document your training data sources here so the data is reproducible across machines:

| Source | URL / path | Size | License |
|--------|-----------|------|---------|
| *(add rows as you collect data)* | | | |

## Preparation Steps

1. Collect `.txt` files and place them in `data/raw/`
2. Train tokenizer: `python -m src.tokenizer.train --data data/raw/ --vocab-size 8192`
3. Tokenize corpus: `python -m src.training.tokenize_corpus`
4. Verify: `python -c "from src.training.dataset import TokenDataset; d = TokenDataset('data/tokenized/', 1024); print(len(d), 'samples')"`
