"""Build the fixed WIR codebooks (wir/codebooks/*.txt).

A codebook maps one word to k numbers. Entries are k-means centroids of standard
normal samples (seeded), so they fit any weight row scaled to unit RMS; no model
data goes in, which is what lets one codebook serve every program. Entries are
sorted by how often Gaussian data picks them and paired, most used first, with
lowercase words that are one token each (with a leading space) in the Qwen2.5
tokenizer, in vocabulary order (roughly by frequency).

    uv run --project ../python-train python scripts/make_codebook.py <qwen tokenizer.json> [--words-only]

`--words-only` keeps the entries of the existing files and only re-pairs words.
Profanity, sexual, violent and identity words are skipped (BLOCK, BLOCK_PREFIX).

Output files are part of the language: regenerating them changes what existing
coded text means, so a changed codebook needs a new name.
"""
import pathlib
import re
import sys

import torch
from tokenizers import Tokenizer

KEYWORDS = {"wir", "input", "output", "crease", "assert", "solve", "over", "hinge", "left", "at", "turn",
            "zero", "on", "line", "plane", "normal", "offset", "through", "between", "and", "equals", "ramp",
            "along", "above", "gate", "silu", "times", "patch", "in", "codebook"}


# Words a reader should not meet in a weight listing.
BLOCK = {"anal", "ass", "cum", "hell", "damn", "gay", "gun", "guns", "rape", "raped", "rapist", "trans", "die", "died",
         "dies", "dead", "death", "blood", "sick", "xxx", "escort", "cancer", "jew", "jews", "abuse", "hate", "ugly",
         "fat", "drunk", "slave", "nig", "kkk", "isis", "hitler", "nazi", "war", "shot", "shoot", "bullet", "crime"}
BLOCK_PREFIX = ("sex", "porn", "fuck", "shit", "cunt", "nigg", "fag", "slut", "whor", "bitch", "dick", "cock", "puss",
                "penis", "vagin", "erot", "masturb", "orgasm", "hentai", "nude", "naked", "kill", "murder", "suicid",
                "terror", "bomb", "weapon", "drug", "cocaine", "heroin", "muslim", "islam", "christ", "lesbian",
                "viol", "tits", "boob", "dildo", "milf", "retard", "gore", "torture", "assault", "weed", "nazi")


def kmeans(x, k, iters, seed):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(x[i:i + 32768], c).argmin(1) for i in range(0, x.shape[0], 32768)])
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k).to(x.dtype)[:, None]
        c = torch.where(n > 0, s / n.clamp(min=1), x[torch.randint(x.shape[0], (k,), generator=g)])
    return c


def words(tok_path, n):
    t = Tokenizer.from_file(tok_path)
    vocab = sorted(t.get_vocab().items(), key=lambda kv: kv[1])
    out = []
    for piece, _ in vocab:
        w = piece[1:]
        if piece.startswith('Ġ') and re.fullmatch('[a-z]{2,8}', w) and w not in KEYWORDS \
                and not re.fullmatch('r[0-9x]*', w) and w not in BLOCK and not w.startswith(BLOCK_PREFIX) and len(t.encode(' ' + w).ids) == 1:
            out.append(w)
        if len(out) == n:
            return out
    raise SystemExit(f'only {len(out)} words')


def build(name, dim, size, samples, tok_path, out_dir, words_only=False):
    path = out_dir / f'{name}.txt'
    if words_only:
        rows = [l.split() for l in path.read_text(encoding='utf-8').splitlines() if l and not l.startswith('#')]
        ws = words(tok_path, size)
        head = path.read_text(encoding='utf-8').splitlines()[0]
        body = '\n'.join(' '.join([w] + r[1:]) for w, r in zip(ws, rows))
        path.write_text(head + '\n' + body + '\n', encoding='utf-8')
        print(f'{path}: words re-paired')
        return
    g = torch.Generator().manual_seed(dim * 1000 + size)
    x = torch.randn(samples, dim, generator=g)
    c = kmeans(x, size, 80, seed=dim)
    test = torch.randn(samples, dim, generator=torch.Generator().manual_seed(7))
    use = torch.bincount(torch.cat([torch.cdist(test[i:i + 32768], c).argmin(1)
                                    for i in range(0, samples, 32768)]), minlength=size)
    c = c[torch.argsort(use, descending=True, stable=True)]
    ws = words(tok_path, size)
    rows = [f'{w} ' + ' '.join(f'{v:.4f}' for v in row.tolist()) for w, row in zip(ws, c)]
    path.write_text(f'# WIR codebook {name}: {size} words x {dim} numbers (scripts/make_codebook.py)\n'
                    + '\n'.join(rows) + '\n', encoding='utf-8')
    err = (test - c[torch.cat([torch.cdist(test[i:i + 32768], c).argmin(1)
                               for i in range(0, samples, 32768)])]).pow(2).mean().sqrt()
    print(f'{path}: rms quantization error on N(0,1) {err:.4f}')


if __name__ == '__main__':
    out = pathlib.Path(__file__).resolve().parent.parent / 'codebooks'
    out.mkdir(exist_ok=True)
    only = '--words-only' in sys.argv
    build('g2', 2, 256, 400_000, sys.argv[1], out, only)
    build('g4', 4, 4096, 800_000, sys.argv[1], out, only)
