from __future__ import annotations

import json
import pathlib
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever


# ── Paths & model ──────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
JSON_FILE  = DATA_DIR / "products_w_tagged.json"
VSTORE_DIR = DATA_DIR / "vector_store"
EMB_MODEL  = "models/paraphrase-multilingual-mpnet-base-v2"   
# ── Utility regex ──────────────────────────────────────────────────────
_MM_RE = re.compile(r"([\d\.,]+)\s*mm", re.I)
def _mm(txt: str | None) -> Optional[float]:
    if not txt: return None
    m = _MM_RE.search(txt)
    if not m:    return None
    return float(m.group(1).replace(",", "."))

def _to_bool(x: Any) -> bool:
    return str(x).strip().lower() == "true"

# ── Flatten technical_data → plain text ────────────────────────────────
def _flat_tech(d: Dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, dict):
            parts.extend(f"{sub_k}: {sub_v}" for sub_k, sub_v in v.items())
        else:
            parts.append(f"{k}: {v}")
    return " ; ".join(parts)

# ── Dimension extractor ────────────────────────────────────────────────
def _dims(prod: dict) -> dict[str, Optional[float]]:
    maße = prod.get("technical_data", {}).get("Maße", {})
    width = height = depth = None
    for k, v in maße.items():
        lk = k.lower()
        if "breite" in lk and width  is None: width  = _mm(str(v))
        if "höhe"   in lk and height is None: height = _mm(str(v))
        if "tiefe"  in lk and depth  is None: depth  = _mm(str(v))
    if width is None and "Rastermaß" in maße:   # fallback
        width = _mm(str(maße["Rastermaß"]))
    return {"width": width, "height": height, "depth": depth}

# ── Load & normalise JSON products ─────────────────────────────────────
def _load() -> List[dict]:
    raw: list[dict] = json.load(JSON_FILE.open(encoding="utf-8"))
    out = []
    for p in raw:
        d = _dims(p)
        out.append({
            "article_number": p["article"],
            "designation"   : p.get("name") or p.get("short_description"),
            "short_description": p.get("short_description", ""),
            "is_rail"       : _to_bool(p.get("is_rail")),
            "product_type"  : p.get("technical_data", {})
                               .get("Artikeleigenschaften", {})
                               .get("Produkttyp"),
            "width" : d["width"], "height": d["height"], "depth": d["depth"],
            "tech_text": _flat_tech(p.get("technical_data", {})),
            "raw": p,
        })
    return out

_PRODUCTS  = _load()
all_rails  = [p for p in _PRODUCTS if p["is_rail"]]
_ART_IDX   = {p["article_number"]: p for p in _PRODUCTS}
_NAME_IDX  = {(p["designation"] or "").lower(): p for p in _PRODUCTS}

# ── Look-up helpers ────────────────────────────────────────────────────
def get_product_by_article(a: str) -> Optional[dict]:
    return _ART_IDX.get(a)

def search_product_by_name(fragment: str, k: int = 5) -> List[dict]:
    frag = fragment.lower()
    return [p for p in _PRODUCTS if frag in (p["designation"] or "").lower()][:k]

# ── Build documents for embedding ──────────────────────────────────────
def _to_doc(prod: dict) -> Document:
    txt = " – ".join(filter(None, [
        prod["article_number"], prod["designation"],
        prod["short_description"], prod["product_type"] or "",
        prod["tech_text"],
    ]))
    return Document(txt, metadata={
        "article_number": prod["article_number"],
        "is_rail": prod["is_rail"],
    })

def _xml_doc(path: pathlib.Path) -> Document:
    return Document(path.read_text(encoding="utf-8"),
                    metadata={"filename": path.name, "is_xml_example": True})

# ── Vector DB build / load ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _vectordb() -> FAISS:
    emb = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
    if VSTORE_DIR.exists():
        return FAISS.load_local(str(VSTORE_DIR), emb,
                                allow_dangerous_deserialization=True)
    docs = [_to_doc(p) for p in tqdm(_PRODUCTS, desc="Embedding products")]
    db = FAISS.from_documents(docs, emb)
    VSTORE_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VSTORE_DIR))
    return db

_vectordb = _vectordb()

# ── Hybrid retriever (BM25 + FAISS) ────────────────────────────────────
_bm25  = BM25Retriever.from_documents(_vectordb.docstore._dict.values())
_bm25.k = 6
_faiss = _vectordb.as_retriever(search_kwargs={"k": 6})
vector_retriever = EnsembleRetriever(retrievers=[_bm25, _faiss],
                                     weights=[0.7, 0.3])

# ── Smart lookup convenience ───────────────────────────────────────────
def _build_line(p: dict) -> str:
    return " – ".join(x for x in
                      [p["article_number"], p["designation"],
                       p["short_description"]] if x)

def smart_lookup(query: str, k: int = 6) -> List[str]:
    q = query.strip()
    if q.isdigit() and q in _ART_IDX:              # exact article
        return [_build_line(_ART_IDX[q])]
    direct = search_product_by_name(q, k)
    if direct:
        return [_build_line(p) for p in direct]
    docs = vector_retriever.get_relevant_documents(q)
    return [d.page_content for d in docs][:k]

# ── Self-test (run `python -m loaders.data_loader`) ────────────────────
if __name__ == "__main__":
    print(f"{len(_PRODUCTS):,} products loaded  |  Rails: {len(all_rails)}")
    for q in ["0804147", "Steckdose Brasilien", "Endklemme"]:
        print(f"\n► QUERY: {q}")
        for line in smart_lookup(q)[:3]:
            print(" ", line[:120])