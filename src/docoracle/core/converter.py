"""AsciiDoc to Markdown conversion module.

Uses regex-based conversion for AsciiDoc to Markdown conversion.
This approach is more reliable for AsciiDoc fragments (which may contain
include directives, etc.) than using pypandoc which expects complete,
resolvable documents.

pypandoc is kept as an optional dependency for potential future use.
"""

import re


def convert_asciidoc_to_markdown(content: str) -> str:
    """Convert AsciiDoc content to Markdown.

    Uses regex-based conversion which works well for AsciiDoc fragments.
    pypandoc may be used in the future for more accurate conversion.

    Args:
        content: Raw AsciiDoc content to convert

    Returns:
        Markdown representation of the content
    """
    # Use regex-based conversion (more reliable for fragments)
    return _convert_asciidoc_to_markdown_regex(content)


def _convert_asciidoc_to_markdown_regex(content: str) -> str:
    """Regex-based AsciiDoc to Markdown conversion fallback.

    Args:
        content: Raw AsciiDoc content to convert

    Returns:
        Markdown representation of the content
    """
    # Convert tables before anything else so the table content survives
    # later transformations (and so we don't munge table internals).
    content = _convert_asciidoc_tables(content)

    # Convert AsciiDoc headings to Markdown
    # AsciiDoc: = Title, == Title, === Title
    # Markdown: # Title, ## Title, ### Title
    content = re.sub(
        r"^(={1,6})\s+(.+?)\s*(?:\[\[(.+?)\]\])?\s*$",
        lambda m: (
            "#" * len(m.group(1))
            + " "
            + m.group(2)
            + (" {#" + m.group(3) + "}" if m.group(3) else "")
        ),
        content,
        flags=re.MULTILINE,
    )

    # Convert bold: ***text*** -> **text**
    content = re.sub(r"\*\*\*(.+?)\*\*\*", r"**\1**", content)

    # Convert code blocks
    # AsciiDoc: [source,language]
    # ----
    # code
    # ----
    # Markdown: ```language
    # code
    # ```
    content = re.sub(
        r"\[source,(\w+)\][ \t]*\n----[ \t]*\n(.*?)\n----[ \t]*",
        r"```\1\n\2\n```",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )

    # Convert block quotes
    # AsciiDoc: ____ text ____
    # Markdown: > text
    content = re.sub(
        r"^____[ \t]*\n(.*?)\n____[ \t]*$", r"> \1", content, flags=re.MULTILINE | re.DOTALL
    )

    # Convert links
    # AsciiDoc: link:url[text] -> [text](url)
    content = re.sub(r"link:(\S+?)\[(.*?)\]", r"[\2](\1)", content)

    # Convert inline links
    # AsciiDoc: https://url[text] -> [text](https://url)
    content = re.sub(r"(\w+://\S+?)\[(.*?)\]", r"[\2](\1)", content)

    # Convert xref macros
    # AsciiDoc: xref:[module::]path/page.adoc[label]  -> [label](page)
    # When the label is empty, fall back to the page name (without extension).
    # Paths may contain spaces (e.g., ``Jahre/Steuer 2024.adoc``), so the
    # target pattern only excludes ``[`` and newline.
    content = re.sub(r"xref:([^\[\n]+?)\[([^\]]*)\]", _convert_xref, content)

    # Convert star-bullet lists to nested Markdown lists.
    #   *   Item   -> - Item
    #   **  Item   ->   - Item
    #   *** Item   ->     - Item
    content = re.sub(
        r"^(\*+)[ \t]+(.+)$",
        lambda m: "  " * (len(m.group(1)) - 1) + "- " + m.group(2),
        content,
        flags=re.MULTILINE,
    )

    # Convert dot-numbered lists to Markdown ordered lists.
    #   . Item     -> 1. Item
    #   .. Item    ->   1. Item
    # Markdown auto-numbers, so we always emit '1.' regardless of position.
    # The space requirement distinguishes list items (``. Foo``) from block
    # titles (``.Foo``).
    content = re.sub(
        r"^(\.+)[ \t]+(\S.*)$",
        lambda m: "  " * (len(m.group(1)) - 1) + "1. " + m.group(2),
        content,
        flags=re.MULTILINE,
    )

    # Drop standalone block anchors like ``[[some-id]]`` on their own line.
    content = re.sub(r"^\[\[[^\]]+\]\][ \t]*$", "", content, flags=re.MULTILINE)

    # Strip AsciiDoc inline-role syntax: ``[.role]#text#`` -> ``text``.
    content = re.sub(r"\[\.[^\]]+\]#([^#\n]+)#", r"\1", content)
    # Strip the bare ``#text#`` mark/highlight syntax similarly.
    content = re.sub(r"(?<!\w)#([^#\n]+)#(?!\w)", r"\1", content)

    # Remove AsciiDoc attributes
    # :name: value
    content = re.sub(r"^:\w+:.+$", "", content, flags=re.MULTILINE)

    # Remove AsciiDoc comments
    content = re.sub(r"^//.*$", "", content, flags=re.MULTILINE)

    # Clean up multiple blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


def _convert_xref(match: "re.Match[str]") -> str:
    """Convert an AsciiDoc xref macro to Markdown link syntax.

    ``xref:module::page.adoc[Label]``    -> ``[Label](page)``
    ``xref:page.adoc[]``                 -> ``[page](page)``
    ``xref:dir/page.adoc[Label]``        -> ``[Label](page)``
    """
    target = match.group(1)
    label = match.group(2).strip()

    # Antora xrefs can be ``page.adoc``, ``module:page.adoc``,
    # ``component::page.adoc``, or ``component::module:page.adoc``. The page is
    # always the last colon-separated segment.
    target = target.rsplit(":", 1)[-1]

    # Use the final path segment as the page name.
    page = target.rsplit("/", 1)[-1]
    # Drop the ``.adoc`` extension if present.
    if page.endswith(".adoc"):
        page = page[:-5]

    if not label:
        label = page

    return f"[{label}]({page})"


def _convert_asciidoc_tables(content: str) -> str:
    """Convert AsciiDoc tables to flattened 'Header: Value' text.

    AsciiDoc tables are delimited by ``|===`` lines, optionally preceded by an
    attribute line like ``[cols="26%,11%,..."]``. Cells start with ``|`` and may
    span multiple source lines (with or without ``+`` continuation).

    The result is one line per data row, each in the form
    ``Header1: Value1 | Header2: Value2 | ...`` — readable as text and friendly
    for embedding, unlike pipe-tables which downstream cleaning strips.
    """
    lines = content.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "|===":
            # Look back for an optional config attribute line.
            cols_spec = ""
            if out and out[-1].strip().startswith("[") and out[-1].strip().endswith("]"):
                config_line = out.pop().strip()
                m = re.search(r'cols\s*=\s*"?([^"\]]+)"?', config_line)
                if m:
                    cols_spec = m.group(1)

            # Find closing delimiter.
            j = i + 1
            while j < len(lines) and lines[j].strip() != "|===":
                j += 1

            body = "\n".join(lines[i + 1 : j])
            flattened = _flatten_table(cols_spec, body)
            if flattened:
                out.append(flattened)
            i = j + 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _flatten_table(cols_spec: str, body: str) -> str:
    """Flatten a table body into 'Header: Value' lines."""
    # Capture the first non-empty body line for column-count inference before
    # we mangle structure.
    first_row_line = next((ln for ln in body.split("\n") if ln.strip()), "")

    # Resolve AsciiDoc line continuation: trailing ` +` joins to next line.
    body = re.sub(r"[ \t]\+\n", " ", body)
    # Cells may span lines; collapse remaining newlines into spaces so we can
    # treat the whole body as one cell-delimited stream.
    body = re.sub(r"\s*\n\s*", " ", body).strip()
    # Strip cell modifier prefixes like ``a|``, ``m|``, ``2+|``, ``^|``, ``<|``, ``>|``.
    # Only match when the modifier is preceded by whitespace, another pipe, or
    # start-of-string — otherwise a German word ending in one of these letters
    # (``Unfall|``, ``und|``) would lose its trailing character.
    body = re.sub(
        r"(^|[\s|])(?:\d+(?:\.\d+)?\+)?[<>^]?[adhmlse]?\|",
        r"\1|",
        body,
    )

    parts = body.split("|")
    # Drop the empty fragment before the very first pipe.
    cells = [p.strip() for p in parts[1:]]
    if not cells:
        return ""

    num_cols = _count_columns(cols_spec)
    if num_cols <= 0:
        # Infer from the first source row of the body — the count of pipes on
        # that line equals the cell count for a well-formed table.
        num_cols = max(first_row_line.count("|"), 1)

    rows = [cells[i : i + num_cols] for i in range(0, len(cells), num_cols)]
    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]

    if not data_rows:
        # Header-only table — emit the headers as a single line.
        return " | ".join(c for c in headers if c)

    lines: list[str] = []
    for row in data_rows:
        pairs: list[str] = []
        for j, val in enumerate(row):
            if not val:
                continue
            h = headers[j] if j < len(headers) and headers[j] else f"Col{j + 1}"
            pairs.append(f"{h}: {val}")
        if pairs:
            lines.append(" | ".join(pairs))

    return "\n".join(lines)


def _count_columns(cols_spec: str) -> int:
    """Parse a cols spec like ``"26%,11%,14%"`` or ``,,`` into a column count."""
    if not cols_spec:
        return 0
    s = cols_spec.strip().strip('"').strip("'")
    return len(s.split(","))
