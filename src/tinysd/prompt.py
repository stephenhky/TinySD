"""Conversion of AUTOMATIC1111 (A1111) prompt syntax into Compel prompt syntax."""

import re


def _convert_a1111_segment_to_compel(segment: str) -> str:
    """Convert A1111 weighting syntax within a single prompt segment to Compel syntax.

    Handles ``<lora:...>`` tag removal, ``[text:weight]``, ``[text]`` (de-emphasis),
    ``(text:weight)`` and bare ``(text)`` (A1111's implicit 1.1x emphasis).

    Args:
        segment: A prompt segment that contains no ``BREAK`` keyword.

    Returns:
        The segment rewritten in Compel syntax.
    """
    # Remove inline A1111 <lora:name:weight> tags so they aren't tokenized as literal words
    segment = re.sub(r"<lora:[^>]+>", "", segment)

    # Convert [text:weight] -> (text)weight
    segment = re.sub(
        r"(?<!\\)\[([^\[\]]+?):\s*([0-9]*\.?[0-9]+)\s*(?<!\\)\]",
        r"(\1)\2",
        segment,
    )

    # Convert [text] (de-emphasis, nested supported) -> (text)-
    prev = None
    while prev != segment:
        prev = segment
        segment = re.sub(r"(?<!\\)\[([^\[\]]+?)(?<!\\)\]", r"(\1)-", segment)

    # Convert (text:weight) -> (text)weight (innermost first to handle nesting)
    prev = None
    while prev != segment:
        prev = segment
        segment = re.sub(
            r"(?<!\\)\(([^()]+?):\s*([0-9]*\.?[0-9]+)\s*(?<!\\)\)",
            r"(\1)\2",
            segment,
        )

    # In A1111, bare (text) multiplies weight by 1.1, whereas in Compel bare (text) is 1.0
    # unless followed by +, -, a weight number, or an operator (.and, .blend, .attend, .swap).
    out = []
    i = 0
    n = len(segment)
    while i < n:
        if segment[i] == ")" and (i == 0 or segment[i - 1] != "\\"):
            out.append(")")
            rest = segment[i + 1 :]
            if not re.match(r"^(?:[+\-]|[0-9]|\.(?:attend|blend|and|add|swap)\b)", rest):
                out.append("+")
            i += 1
        else:
            out.append(segment[i])
            i += 1
    return "".join(out)


def convert_a1111_to_compel(prompt: str) -> str:
    """Convert AUTOMATIC1111 prompt syntax (weights, brackets, BREAK) into Compel syntax.

    Prompts that already use Compel's top-level ``.and()``/``.add()``/``.blend()``
    are returned unchanged. Prompts containing ``BREAK`` are split into chunks and
    combined with Compel's ``.and()`` conjunction.

    Args:
        prompt: A prompt written in A1111 syntax.

    Returns:
        The equivalent prompt in Compel syntax. Empty or whitespace-only prompts are
        returned as-is.

    Examples:
        >>> convert_a1111_to_compel("a (cat:1.3) on a [mat]")
        'a (cat)1.3 on a (mat)-'
    """
    if not prompt or not prompt.strip():
        return prompt
    # Leave prompts that already use Compel's top-level .and()/.blend() untouched
    if re.search(r"\)\s*\.(?:and|add|blend)\s*\(", prompt):
        return prompt

    chunks = re.split(r"\bBREAK\b", prompt)
    converted_chunks = [_convert_a1111_segment_to_compel(c).strip() for c in chunks]
    if len(converted_chunks) == 1:
        return converted_chunks[0]

    escaped_chunks = [c.replace('"', '\\"') for c in converted_chunks]
    joined = ", ".join(f'"{c}"' for c in escaped_chunks)
    return f"({joined}).and()"
