#!/usr/bin/env python3
"""
Extract 4 corner points of a quadrilateral from an fp_poly in a KiCad .kicad_mod file.

Supports:
- fp_poly with exactly 4 (xy ...) points
- fp_poly with 5 points where the last repeats the first

Usage:
  python extract_fp_poly_quad.py path/to/footprint.kicad_mod
"""

from __future__ import annotations
import sys
from typing import Any, List, Tuple, Optional

Point = Tuple[float, float]


# ----------------------------
# Minimal S-expression parser
# ----------------------------

def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if c.isspace():
            i += 1
            continue

        if c == ';':  # KiCad line comment
            while i < n and text[i] != '\n':
                i += 1
            continue

        if c in ("(", ")"):
            tokens.append(c)
            i += 1
            continue

        if c == '"':  # quoted string
            i += 1
            buf = []
            while i < n:
                if text[i] == '"' and text[i - 1] != '\\':
                    break
                buf.append(text[i])
                i += 1
            tokens.append('"' + ''.join(buf) + '"')
            i += 1
            continue

        # atom
        start = i
        while i < n and (not text[i].isspace()) and text[i] not in ("(", ")"):
            i += 1
        tokens.append(text[start:i])

    return tokens


def parse(tokens: List[str]) -> Any:
    def atom(tok: str) -> Any:
        if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
            return tok[1:-1]
        try:
            return float(tok)
        except ValueError:
            return tok

    stack: List[List[Any]] = []
    cur: List[Any] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok == "(":
            stack.append(cur)
            new_list: List[Any] = []
            cur.append(new_list)
            cur = new_list
        elif tok == ")":
            if not stack:
                raise ValueError("Unbalanced ')'")
            cur = stack.pop()
        else:
            cur.append(atom(tok))
        i += 1

    if stack:
        raise ValueError("Unbalanced '('")

    # Most KiCad files parse to a single root list
    return cur[0] if len(cur) == 1 else cur


# ----------------------------
# fp_poly quad extraction
# ----------------------------

def is_list(x: Any) -> bool:
    return isinstance(x, list)


def walk(node: Any):
    if is_list(node):
        yield node
        for child in node:
            yield from walk(child)


def head(node: Any) -> Optional[str]:
    return node[0] if is_list(node) and node and isinstance(node[0], str) else None


def extract_fp_poly_quad(tree: Any) -> Point | List[Point]:
    for node in walk(tree):
        if head(node) != "fp_poly":
            continue

        # Find (pts (xy x y) (xy x y) ...)
        pts_node = None
        for child in node:
            if is_list(child) and head(child) == "pts":
                pts_node = child
                break
        if pts_node is None:
            continue

        pts: List[Point] = []
        for child in pts_node[1:]:
            if is_list(child) and head(child) == "xy" and len(child) >= 3:
                pts.append((float(child[1]), float(child[2])))

        if len(pts) == 4:
            return pts

        if len(pts) == 5 and pts[0] == pts[-1]:
            return pts[:-1]

    raise RuntimeError(
        "No quadrilateral fp_poly found. Expected fp_poly->pts with 4 points, "
        "or 5 points where last equals first."
    )

def quad_extract(file):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    tree = parse(tokenize(text))
    pts = extract_fp_poly_quad(tree)
    return pts


def main(argv: List[str]) -> int:
    path = "footprints.pretty/coil.kicad_mod"
    pts = quad_extract(path)
    print(pts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

