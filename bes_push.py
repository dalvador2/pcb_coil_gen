from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

Point = Tuple[float, float]


# ----------------------------
# Preserve KiCad string vs symbol
# ----------------------------

@dataclass(frozen=True)
class KStr:
	"""Represents a quoted KiCad string token."""
	value: str


# ----------------------------
# Minimal S-expression parser
# ----------------------------

def _tokenize(text: str) -> List[str]:
	tokens: List[str] = []
	i = 0
	n = len(text)

	while i < n:
		c = text[i]

		if c.isspace():
			i += 1
			continue

		if c == ';':  # comment to end-of-line
			while i < n and text[i] != '\n':
				i += 1
			continue

		if c in ("(", ")"):
			tokens.append(c)
			i += 1
			continue

		if c == '"':  # quoted string
			i += 1
			buf: List[str] = []
			while i < n:
				if text[i] == '"' and text[i - 1] != '\\':
					break
				buf.append(text[i])
				i += 1
			# store WITH quotes so we can recognize it later
			tokens.append('"' + "".join(buf) + '"')
			i += 1
			continue

		# atom
		start = i
		while i < n and (not text[i].isspace()) and text[i] not in ("(", ")"):
			i += 1
		tokens.append(text[start:i])

	return tokens


def _parse(tokens: List[str]) -> Any:
	def atom(tok: str) -> Any:
		# quoted string -> KStr
		if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
			inner = tok[1:-1]
			# unescape \" and \\ minimally
			inner = inner.replace('\\"', '"').replace('\\\\', '\\')
			return KStr(inner)

		# number -> float
		try:
			return float(tok)
		except ValueError:
			# symbol -> plain str
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

	return cur[0] if len(cur) == 1 else cur


# ----------------------------
# S-expression writer (TABS) + preserves quoted strings
# ----------------------------

def _fmt_atom(x: Any) -> str:
	if isinstance(x, KStr):
		s = x.value.replace("\\", "\\\\").replace('"', '\\"')
		return f"\"{s}\""

	if isinstance(x, str):
		# symbol
		return x

	if isinstance(x, float):
		s = f"{x:.6f}".rstrip("0").rstrip(".")
		return s if s else "0"

	return str(x)


def _dumps_sexpr(node: Any, indent: int = 0) -> str:
	tab = "\t" * indent

	if not isinstance(node, list):
		return tab + _fmt_atom(node)

	if not node:
		return tab + "()"

	# Keep small, atom-only lists in one line (looks like KiCad)
	if all(not isinstance(ch, list) for ch in node) and len(node) <= 8:
		inner = " ".join(_fmt_atom(ch) for ch in node)
		return tab + f"({inner})"

	lines: List[str] = []
	lines.append(tab + "(" + _fmt_atom(node[0]))

	for ch in node[1:]:
		if isinstance(ch, list):
			lines.append(_dumps_sexpr(ch, indent + 1))
		else:
			lines.append("\t" * (indent + 1) + _fmt_atom(ch))

	lines.append(tab + ")")
	return "\n".join(lines)


# ----------------------------
# Tree helpers + quad finder (fp_poly only)
# ----------------------------

def _is_list(x: Any) -> bool:
	return isinstance(x, list)


def _head(node: Any) -> Optional[str]:
	return node[0] if _is_list(node) and node and isinstance(node[0], str) else None


def _walk_with_parent(node: Any, parent: Optional[list] = None):
	if _is_list(node):
		yield node, parent
		for child in node:
			yield from _walk_with_parent(child, node)


def _find_first_quad_fp_poly(tree: Any) -> Tuple[list, int]:
	for node, parent in _walk_with_parent(tree):
		if _head(node) != "fp_poly" or parent is None:
			continue

		pts_node = None
		for child in node:
			if _is_list(child) and _head(child) == "pts":
				pts_node = child
				break
		if pts_node is None:
			continue

		pts: List[Point] = []
		for child in pts_node[1:]:
			if _is_list(child) and _head(child) == "xy" and len(child) >= 3:
				pts.append((float(child[1]), float(child[2])))

		if len(pts) == 4 or (len(pts) == 5 and pts[0] == pts[-1]):
			for idx, ch in enumerate(parent):
				if ch is node:
					return parent, idx

	raise RuntimeError(
		"No quadrilateral fp_poly found (needs 4 points, or 5 with last == first)."
	)


def _detect_stroke_style(tree: Any) -> bool:
	for node, _ in _walk_with_parent(tree):
		if _head(node) == "stroke":
			return True
	return False


# ----------------------------
# Bezier creation
# ----------------------------

def _chunk4(points: List[Point]) -> List[Tuple[Point, Point, Point, Point]]:
	if len(points) % 4 != 0:
		raise ValueError(f"Point list length must be a multiple of 4; got {len(points)}.")
	out = []
	for i in range(0, len(points), 4):
		p1, h1, h2, p2 = points[i:i + 4]
		out.append((p1, h1, h2, p2))
	return out


def _make_fp_curve(p1: Point, h1: Point, h2: Point, p2: Point, width: float, use_stroke: bool) -> list:
	pts = [
		"pts",
		["xy", p1[0], p1[1]],
		["xy", h1[0], h1[1]],
		["xy", h2[0], h2[1]],
		["xy", p2[0], p2[1]],
	]
	layer = ["layer", KStr("F.Cu")]

	if use_stroke:
		# KiCad v7+ style
		stroke = ["stroke", ["width", float(width)], ["type", "solid"]]
		return ["fp_curve", pts, stroke, layer]

	# KiCad v6 style
	return ["fp_curve", pts, layer, ["width", float(width)]]


# ----------------------------
# Public function you asked for
# ----------------------------

def add_fcu_beziers_after_quad_fp_poly(
	width: float,
	points: List[Point],
	input_filepath: str,
	output_filepath: str,
) -> None:
	"""
	- Finds the first quadrilateral fp_poly in input_filepath
	- Appends fp_curve Bezier segments (p1, h1, h2, p2) on F.Cu
	- Writes result to output_filepath

	points: flat list in blocks of 4: [p1, h1, h2, p2, p1, h1, h2, p2, ...]
	"""
	if width <= 0:
		raise ValueError("width must be > 0")

	if not isinstance(points, list) or not points:
		raise ValueError("points must be a non-empty list of (x, y) tuples")

	for p in points:
		if not (isinstance(p, (tuple, list)) and len(p) == 2):
			raise ValueError("points must be a list of (x, y) tuples")
		if not (isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))):
			raise ValueError("points must contain numeric (x, y) values")

	with open(input_filepath, "r", encoding="utf-8") as f:
		text = f.read()

	tree = _parse(_tokenize(text))
	parent, idx = _find_first_quad_fp_poly(tree)
	use_stroke = _detect_stroke_style(tree)

	segs = _chunk4([(float(x), float(y)) for (x, y) in points])
	curves = [_make_fp_curve(p1, h1, h2, p2, width=width, use_stroke=use_stroke) for (p1, h1, h2, p2) in segs]

	insert_at = idx + 1
	for c in curves:
		parent.insert(insert_at, c)
		insert_at += 1

	out_text = _dumps_sexpr(tree) + "\n"
	with open(output_filepath, "w", encoding="utf-8") as f:
		f.write(out_text)

