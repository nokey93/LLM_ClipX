import re
from typing import Dict, List, Optional

from loaders import data_loader

SEG_RE = re.compile(r",|\band\b|\bwith\b", re.I)
ART_RE = re.compile(r"\b(\d{6,})\b")
QTY_RE = re.compile(r"\b(\d+)\s*(?:x|X)?")


def parse_instruction(text: str) -> List[Dict[str, Optional[str]]]:
    """Parse a free-text order instruction into items.

    Returns a list of dictionaries with quantity, article number and name.
    The parser is heuristic but works reasonably well for short instructions
    like "Need 1 rail 0801733 cut to 300mm with 5x 0711344 and 1 end clamp 1201442".
    """
    items = []
    for seg in SEG_RE.split(text):
        seg = seg.strip()
        if not seg:
            continue
        qty = 1
        m_qty = QTY_RE.search(seg)
        if m_qty:
            qty = int(m_qty.group(1))
        article = None
        m_art = ART_RE.search(seg)
        if m_art:
            article = m_art.group(1)
        name = seg
        if m_qty:
            name = seg[:m_qty.start()] + seg[m_qty.end():]
        if article:
            name = name.replace(article, "")
        name = re.sub(r"\s+", " ", name).strip()
        items.append({"quantity": qty, "article": article, "name": name})
    return items


def lookup_items(parsed: List[Dict[str, Optional[str]]]) -> List[str]:
    """Return human readable lookup results for parsed items."""
    results = []
    for itm in parsed:
        if itm["article"]:
            results.append(f"{itm['quantity']} x article {itm['article']}")
        else:
            results.append(f"{itm['quantity']} x {itm['name']}")
    return results


def search_items(parsed: List[Dict[str, Optional[str]]], k: int = 3) -> List[List[str]]:
    """Lookup each parsed item in the vector database.

    Each entry in the returned list corresponds to one input item and
    contains the top-k results from :func:`data_loader.smart_lookup`.
    """
    out = []
    for itm in parsed:
        query = itm.get("article") or itm.get("name")
        out.append(data_loader.smart_lookup(query, k=k))
    return out


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:])
    if not query:
        query = "Need 1 rail 0801733 cut to 300mm with 5x 0711344 and 1 end clamp 1201442"
    items = parse_instruction(query)
    lookups = search_items(items)
    for itm, res in zip(items, lookups):
        print(f"{itm['quantity']} × {itm.get('article') or itm['name']}")
        for line in res:
            print(' ', line)