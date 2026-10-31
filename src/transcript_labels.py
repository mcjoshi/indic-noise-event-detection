"""Transcript-conditioned pseudo-labels: every Vaani transcript marks which noise events occur and where among the
words (`<horn> </horn>` between words, `<insect noise> ... </insect noise>` around them). On Gold the non-structural
tag count equals the verified event count on 95.8% of val clips. This turns teacher candidate spans into labels that
respect the transcript: one span per event tag, matched by position (tag's relative character offset in the tag-free
text as a time prior) and teacher support; a tag with no candidate gets a span at its prior position.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.optimize import linear_sum_assignment

NONEV = {"<noise>", "<PAUSE>", "<pause>", "<static noise>", "<talking>", "[unintelligible]", "<people talking>", "[inaudible]", "[noise]", "[talking]"}
TAG = re.compile(r"\[[^\]]+\]|</?[^>]+>")


def event_tags(transcript: str) -> list[tuple[str, float, float]]:
    """(tag, start_frac, end_frac) per event tag, fractions of the tag-free text length; point tags have start == end."""
    t = re.sub(r"\{[^}]*\}", "", transcript or "")
    plain, pos, opened, out = [], 0, {}, []
    i = 0
    for m in TAG.finditer(t):
        seg = t[i:m.start()]; plain.append(seg); pos += len(seg); i = m.end()
        tok = m.group(0)
        if tok.startswith("["):
            if tok not in NONEV: out.append((tok, pos, pos))
        elif tok.startswith("</"):
            name = "<" + tok[2:]
            if name in opened:
                s = opened.pop(name)
                if name not in NONEV: out.append((name, s, pos))
        else:
            opened[tok] = pos
    plain.append(t[i:]); total = max(1, len("".join(plain)))
    for name, s in opened.items():
        if name not in NONEV: out.append((name, s, total))
    return [(n, s / total, e / total) for n, s, e in out]


def label(transcript: str, dur: float, cands: list[tuple[float, float, int]], max_support: int,
          w_pos: float = 1.0, fill: bool = True, fill_len: float = 0.45) -> list[tuple[float, float]]:
    """cands: (onset, offset, support). One span per event tag by Hungarian matching on position and support."""
    tags = event_tags(transcript)
    if not tags:
        return []
    if not cands:
        return [(max(0.0, (s + e) / 2 * dur - fill_len / 2), min(dur, (s + e) / 2 * dur + fill_len / 2)) for _, s, e in tags] if fill else []
    C = np.zeros((len(tags), len(cands)))
    for i, (_, s, e) in enumerate(tags):
        tc = (s + e) / 2 * dur
        for j, (a, b, sup) in enumerate(cands):
            C[i, j] = w_pos * abs((a + b) / 2 - tc) / max(dur, 1e-3) - sup / max_support
    r, c = linear_sum_assignment(C)
    out = [(cands[j][0], cands[j][1]) for j in c]
    if fill:
        for i in set(range(len(tags))) - set(r):
            _, s, e = tags[i]; tc = (s + e) / 2 * dur
            out.append((max(0.0, tc - fill_len / 2), min(dur, tc + fill_len / 2)))
    return sorted(out)
