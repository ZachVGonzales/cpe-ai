#!/usr/bin/env python
# filepath: c:\Users\Zachv\Documents\svc-ai\scripts\data_ingestion\data_prep.py

"""
Data Preparation Script for RAG Vector Store (v2.1)

Major upgrades focused on QUALITY over speed:
1) Layout-aware parsing + normalization
   - Prefer pdfplumber over pypdf; optional OCR hook
   - Unicode normalization (NFKC), de-hyphenation, ligature fixes, smart newline collapse
   - Readability-first HTML extraction path kept
2) Semantic boundary detection (similarity valley + adaptive overlap)
   - When near token budget, use sentence/paragraph similarity to pick a natural split
   - Overlap adapts: 10% at strong boundaries, 20–25% at weak ones
3) Code/Table special handling
   - Treat fenced code and Markdown tables as atomic blocks
   - Never break mid-item; summarize table headers into chunk context
4) Lean/Mathlib API Documentation Support (NEW in v2.1)
   - Specialized LeanMarkdownProcessor for Lean theorem prover API docs
   - Preserves source markers, theorem definitions, and type signatures together
   - Recognizes Lean-specific patterns (source, theorem, def, lemma, etc.)
   - Keeps definitions atomic to avoid breaking semantic units
   - Minimal filtering to preserve terse but important content
   - Module name tracking for better context breadcrumbs

CLI is unchanged; defaults remain tokens-based (not characters).

NOTE: Optional dependencies are used when available and gracefully degrade when missing.
- pdfplumber (preferred PDF text extractor)
- scikit-learn (for TF-IDF boundary scoring)

"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlparse
import json
import chardet
from tqdm import tqdm
import time
from urllib.robotparser import RobotFileParser
import sys
import unicodedata
from dataclasses import dataclass

# Add project root to path for imports when running as script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from transformers import AutoTokenizer 
from src.cpe_ai.config.settings import DEFAULT_QA_MODEL
QA_TOKEN = None

# Optional PDF & ML libs
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

# For web scraping
import requests
from bs4 import BeautifulSoup
import lxml  # noqa: F401

# For document processing
import pypdf

# For DOCX processing (optional)
try:
    from docx import Document as DocxDocument  # type: ignore
except ImportError:
    DocxDocument = None  # type: ignore

# For text chunking
import nltk
from nltk.tokenize import sent_tokenize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "../../vision-vs/processed-docs")
INDEX_FILENAME = "document_index.json"
DEFAULT_CHUNK_SIZE = 800  # tokens
DEFAULT_CHUNK_OVERLAP = 120  # tokens (~15%)
DEFAULT_MIN_WORDS = 25
MAX_ALLOWED_OVERLAP_RATIO = 0.25
SOFT_OVERFLOW_RATIO = 0.12  # allow up to +12% at natural boundary

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# Global tokenizer cache
_MODEL_TOKENIZER = None

def get_model_tokenizer():
    global _MODEL_TOKENIZER
    if _MODEL_TOKENIZER is not None:
        return _MODEL_TOKENIZER
    if AutoTokenizer is None or DEFAULT_QA_MODEL is None:
        raise RuntimeError("Transformers AutoTokenizer or DEFAULT_QA_MODEL not available.")
    logger.info(f"Loading tokenizer for token-based chunking: {DEFAULT_QA_MODEL}")
    try:
        _MODEL_TOKENIZER = AutoTokenizer.from_pretrained(
            DEFAULT_QA_MODEL,
            token=QA_TOKEN or None,
            use_fast=True
        )
        return _MODEL_TOKENIZER
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to load tokenizer for {DEFAULT_QA_MODEL}: {e}")
        raise


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tokenizer = get_model_tokenizer()
    return len(tokenizer(text, add_special_tokens=False).input_ids)


# ---------------------------
# Normalization utilities
# ---------------------------

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    # Replace common ligatures
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    # De-hyphenate line breaks: exam-\nple -> example
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse excessive spaces and tidy newlines
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


# ---------------------------
# Document processors
# ---------------------------

class DocumentProcessor:
    def extract_text(self, file_path: str) -> str:
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        # Basic artifacts cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        # Remove long non-word garbage
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\{\}\'"\?\/\\\&\%\$\#\@\!\*\+\=]{5,}', ' ', text)
        text = text.replace('\t', ' ')
        # Strip lines and drop empties
        lines = [ln.strip() for ln in text.split('\n')]
        lines = [ln for ln in lines if ln]
        text = '\n'.join(lines).strip()
        # NEW: normalize
        return normalize_text(text)

    def filter_semantic_content(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split('\n')
        filtered = []
        in_toc = False
        toc_consec = 0
        dotted = re.compile(r'^.*\.{3,}.*\d+\s*$')
        page_num = re.compile(r'^\s*\d+\s*$')
        header_footer = re.compile(r'^\s*(Page \d+|[A-Z][a-z]+ \d{4}|[A-Z\s\.\-]+)$')
        section_header = re.compile(r'^\s*\d+(\.\d+)*\s+.+$')
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            is_dot = bool(dotted.match(s))
            is_pn = bool(page_num.match(s))
            is_hf = bool(header_footer.match(s))
            is_sec = bool(section_header.match(s))
            if is_dot:
                toc_consec += 1
                if toc_consec >= 3:
                    in_toc = True
            else:
                if toc_consec > 0 and not (is_pn or is_hf):
                    toc_consec = 0
                    if in_toc:
                        in_toc = False
            if in_toc or is_pn or is_hf:
                continue
            # Heuristic: skip likely TOC-style headers
            if is_sec and i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if dotted.match(nxt) or page_num.match(nxt):
                    continue
            filtered.append(s)
        return '\n'.join(filtered)

# PDF (layout-aware preferred)
class PDFProcessor(DocumentProcessor):
    def extract_text(self, file_path: str) -> str:
        try:
            logger.info(f"Processing PDF: {file_path}")
            text = ""
            if pdfplumber is not None:
                with pdfplumber.open(file_path) as pdf:
                    parts = []
                    for p in pdf.pages:
                        parts.append(p.extract_text(layout=True) or "")
                    text = "\n".join(parts)
            else:
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        text += (page.extract_text() or "") + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return ""

class DocxProcessor(DocumentProcessor):
    def extract_text(self, file_path: str) -> str:
        try:
            logger.info(f"Processing DOCX: {file_path}")
            if DocxDocument is None:
                logger.error("python-docx library not available. Install with: pip install python-docx")
                return ""
            doc = DocxDocument(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            return ""

class TextProcessor(DocumentProcessor):
    def extract_text(self, file_path: str) -> str:
        try:
            logger.info(f"Processing text file: {file_path}")
            with open(file_path, 'rb') as fh:
                raw = fh.read()
                detected = chardet.detect(raw)
                encoding = detected['encoding'] or 'utf-8'
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return ""

class WebPageProcessor(DocumentProcessor):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Linux; Ubuntu) RAG-DataPrep/1.0'})

    def extract_text(self, url: str) -> str:
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Processing web page: {url} (attempt {attempt + 1}/{self.max_retries})")
                try:
                    rp = RobotFileParser()
                    rp.set_url(urlparse(url).scheme + "://" + urlparse(url).netloc + "/robots.txt")
                    rp.read()
                    if not rp.can_fetch("*", url):
                        logger.warning(f"Blocked by robots.txt: {url}")
                        return ""
                except Exception:
                    logger.debug("Could not check robots.txt; proceeding")
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                if 'text/html' not in r.headers.get('content-type', '').lower():
                    logger.warning(f"Non-HTML content type for {url}")
                    return ""
                soup = BeautifulSoup(r.content, 'lxml')
                for tag in ["script", "style", "header", "footer", "nav", "aside", "menu", "menuitem", "button", "input", "form", "noscript", "iframe", "object", "embed"]:
                    for el in soup.find_all(tag):
                        el.decompose()
                for cls in ['advertisement', 'ads', 'sidebar', 'menu', 'navigation', 'social', 'share', 'comment', 'footer', 'header']:
                    for el in soup.find_all(class_=lambda x: x and cls in x.lower()):
                        el.decompose()
                main = None
                for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main']:
                    main = soup.select_one(selector)
                    if main:
                        break
                text = (main.get_text() if main else soup.get_text())
                text = self._clean_web_text(text)
                return text if text.strip() else ""
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error for {url} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"Error processing web page {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        logger.error(f"Failed to extract text from {url} after {self.max_retries} attempts")
        return ""

    def _clean_web_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        lines = [ln.strip() for ln in text.split('\n')]
        lines = [ln for ln in lines if ln and not (len(ln.split()) == 1 and ln.isupper() and len(ln) < 15)]
        return '\n'.join(lines)

class HTMLFileProcessor(DocumentProcessor):
    def extract_text(self, file_path: str) -> str:
        try:
            logger.info(f"Processing HTML file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
            if not html.strip():
                return ""
            soup = BeautifulSoup(html, 'lxml')
            for tag in ["script", "style", "header", "footer", "nav", "aside", "menu", "menuitem", "button", "input", "form", "noscript", "iframe", "object", "embed"]:
                for el in soup.find_all(tag):
                    el.decompose()
            for cls in ['advertisement', 'ads', 'sidebar', 'menu', 'navigation', 'social', 'share', 'comment', 'footer', 'header']:
                for el in soup.find_all(class_=lambda x: x and cls in x.lower()):
                    el.decompose()
            main = None
            for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main']:
                main = soup.select_one(selector)
                if main:
                    break
            text = (main.get_text() if main else soup.get_text())
            return WebPageProcessor()._clean_web_text(text)
        except Exception as e:
            logger.error(f"Error processing HTML file {file_path}: {e}")
            return ""


class LeanMarkdownProcessor(TextProcessor):
    """Specialized processor for Lean API documentation markdown files.
    
    These files have a specific structure with source markers, theorem definitions,
    type signatures, and code blocks that should be preserved together.
    """
    
    def extract_text(self, file_path: str) -> str:
        try:
            logger.info(f"Processing Lean markdown file: {file_path}")
            # Use parent class to read the file
            text = super().extract_text(file_path)
            if not text:
                return ""
            
            # Preserve Lean-specific structure
            # Don't do aggressive cleaning that might break definitions
            return text
        except Exception as e:
            logger.error(f"Error processing Lean markdown file {file_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Minimal cleaning for Lean markdown to preserve structure."""
        if not text or not text.strip():
            return ""
        
        # Only remove truly problematic characters, preserve structure
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Normalize but don't aggressively collapse newlines
        # (Lean definitions often have specific formatting)
        text = normalize_text(text)
        
        return text.strip()
    
    def filter_semantic_content(self, text: str) -> str:
        """For Lean docs, we want to keep most content including short entries."""
        # Don't filter TOC-like content since Lean docs are structured differently
        # Just remove obvious page numbers
        lines = text.split('\n')
        filtered = []
        page_num = re.compile(r'^\s*Page\s+\d+\s*$')
        
        for line in lines:
            if page_num.match(line.strip()):
                continue
            filtered.append(line)
        
        return '\n'.join(filtered)


# ---------------------------
# Semantic-aware chunker (with boundary scoring)
# ---------------------------

HEADING_MD = re.compile(r'^(#{1,6})\s+(.+)$')
HEADING_NUM = re.compile(r'^\s*(\d+(?:\.\d+){0,5})\s+(.+)$')
LIST_BULLET = re.compile(r'^\s*([\-*+])\s+')
LIST_NUM = re.compile(r'^\s*(\d+[\.\)])\s+')
CHECKBOX = re.compile(r'^\s*[\-\*]\s+\[(?: |x|X)\]\s+')
CODE_FENCE = re.compile(r'^\s*```')
TABLE_ROW = re.compile(r'^\s*\|.+\|\s*$')
TABLE_SEP = re.compile(r'^\s*\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|\s*$')

# Lean-specific patterns
LEAN_SOURCE_MARKER = re.compile(r'^source(?:@\[.+?\])?\s*$')
LEAN_THEOREM = re.compile(r'^(theorem|lemma|def|instance|class|structure|inductive|axiom|example)\s+')
LEAN_TYPE_SIG = re.compile(r'^\s*[:\{]')  # Lines that look like type signatures

@dataclass
class Block:
    type: str  # 'heading', 'paragraph', 'list', 'code', 'table', 'lean_def'
    text: str
    level: int = 0
    metadata: dict = None  # For storing additional context (e.g., definition name)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def is_boundary(self) -> bool:
        return self.type in ('paragraph', 'list', 'code', 'table', 'lean_def')

class SimilarityGate:
    """Cheap sentence/paragraph similarity using TF-IDF if available, else bag-of-words.
    Used to decide whether to split at current boundary or pack one more unit.
    """
    def __init__(self):
        self.vectorizer = None
        if TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=5000)

    def score(self, left: List[str], right: List[str]) -> float:
        # Return cosine similarity between concatenated left and right (0..1). Higher = more similar.
        l = " ".join(left).strip() or "_"
        r = " ".join(right).strip() or "_"
        if self.vectorizer is not None and cosine_similarity is not None:
            try:
                X = self.vectorizer.fit_transform([l, r])
                # cosine_similarity on two 1xN sparse matrices returns a 1x1 array.
                # Casting that array directly to float triggers a NumPy deprecation warning.
                # Explicitly extract the scalar value.
                sim_mat = cosine_similarity(X[0], X[1])  # shape (1,1)
                sim = sim_mat[0, 0]
                return float(sim)
            except Exception:
                pass
        # Fallback: Jaccard over word sets (rough but stable)
        wl = set(re.findall(r"\w+", l.lower()))
        wr = set(re.findall(r"\w+", r.lower()))
        if not wl and not wr:
            return 1.0
        inter = len(wl & wr)
        union = len(wl | wr)
        return inter / max(1, union)

class SemanticChunker:
    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int, breadcrumb_max_depth: int = 3):
        self.tok = get_model_tokenizer()
        self.size = chunk_size_tokens
        self.base_overlap = chunk_overlap_tokens
        self.soft_limit = int(self.size * (1 + SOFT_OVERFLOW_RATIO))
        self.breadcrumb_max_depth = breadcrumb_max_depth
        self.sim_gate = SimilarityGate()

    # Stage 1: blockify
    def blockify(self, text: str) -> List[Block]:
        lines = text.split('\n')
        blocks: List[Block] = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            
            # Code fences
            if CODE_FENCE.match(ln):
                code = [ln]
                lang = ln.strip().replace('```', '').strip() or 'code'
                i += 1
                while i < len(lines):
                    code.append(lines[i])
                    if CODE_FENCE.match(lines[i]):
                        i += 1
                        break
                    i += 1
                blocks.append(Block('code', '\n'.join(code), metadata={'language': lang}))
                continue
            
            # Lean-specific: source markers followed by definitions
            if LEAN_SOURCE_MARKER.match(ln.strip()):
                # Collect the definition that follows
                lean_def = [ln]
                j = i + 1
                def_name = None
                # Look ahead for the actual definition
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt:  # empty line
                        j += 1
                        continue
                    # Check if this is a theorem/def/etc
                    m = LEAN_THEOREM.match(nxt)
                    if m:
                        def_name = self._extract_lean_name(lines[j:min(j+5, len(lines))])
                        # Collect the full definition (until we hit another source or heading or blank lines)
                        while j < len(lines):
                            nxt_line = lines[j]
                            lean_def.append(nxt_line)
                            j += 1
                            # Stop at next source marker, heading, or double blank
                            if (j < len(lines) and 
                                (LEAN_SOURCE_MARKER.match(lines[j].strip()) or 
                                 HEADING_MD.match(lines[j]) or
                                 (not nxt_line.strip() and j < len(lines) and not lines[j].strip()))):
                                break
                        break
                    else:
                        # Not a definition line, might be other content
                        lean_def.append(lines[j])
                        j += 1
                        if j >= len(lines) or HEADING_MD.match(lines[j]) or LEAN_SOURCE_MARKER.match(lines[j].strip()):
                            break
                
                blocks.append(Block('lean_def', '\n'.join(lean_def), metadata={'name': def_name}))
                i = j
                continue
            
            # Tables
            if TABLE_ROW.match(ln):
                tbl = [ln]
                j = i + 1
                has_sep = False
                while j < len(lines) and (TABLE_ROW.match(lines[j]) or TABLE_SEP.match(lines[j])):
                    if TABLE_SEP.match(lines[j]):
                        has_sep = True
                    tbl.append(lines[j])
                    j += 1
                if has_sep:
                    blocks.append(Block('table', '\n'.join(tbl)))
                    i = j
                    continue
            
            # Markdown headings
            m = HEADING_MD.match(ln)
            if m:
                level = len(m.group(1)); title = m.group(2).strip()
                blocks.append(Block('heading', title, level))
                i += 1; continue
            
            # Numbered headings
            n = HEADING_NUM.match(ln)
            if n and len(ln.strip().split()) <= 25:
                level = min(6, n.group(1).count('.') + 1)
                blocks.append(Block('heading', n.group(0).strip(), level))
                i += 1; continue
            
            # Lists
            if LIST_BULLET.match(ln) or LIST_NUM.match(ln) or CHECKBOX.match(ln):
                L = [ln]
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if (LIST_BULLET.match(nxt) or LIST_NUM.match(nxt) or CHECKBOX.match(nxt) or
                        (nxt.startswith('  ') and not (HEADING_MD.match(nxt) or CODE_FENCE.match(nxt)))):
                        L.append(nxt); j += 1
                    else:
                        break
                blocks.append(Block('list', '\n'.join(L)))
                i = j; continue
            
            # Paragraphs
            if ln.strip():
                para = [ln]
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if (not nxt.strip() or HEADING_MD.match(nxt) or HEADING_NUM.match(nxt) or CODE_FENCE.match(nxt) or
                        LIST_BULLET.match(nxt) or LIST_NUM.match(nxt) or CHECKBOX.match(nxt) or TABLE_ROW.match(nxt) or
                        LEAN_SOURCE_MARKER.match(nxt.strip())):
                        break
                    para.append(nxt); j += 1
                blocks.append(Block('paragraph', '\n'.join(para)))
                i = j; continue
            i += 1
        return blocks
    
    def _extract_lean_name(self, lines: List[str]) -> str:
        """Extract the name from a Lean definition."""
        for line in lines:
            m = LEAN_THEOREM.match(line.strip())
            if m:
                # Extract name after the keyword
                rest = line.strip()[len(m.group(0)):]
                # Name is typically before : or {
                name_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_\.]*)', rest)
                if name_match:
                    return name_match.group(1)
        return None

    def _sentences(self, text: str) -> List[str]:
        # Be friendlier to bullet continuations
        safe = text.replace('\n• ', '. ').replace('\n- ', '. ').replace('\n* ', '. ')
        return [s.strip() for s in sent_tokenize(safe) if s.strip()]

    # Stage 2: pack with boundary scoring
    def pack(self, blocks: List[Block]) -> List[str]:
        chunks: List[str] = []
        curr: List[str] = []
        curr_tokens = 0
        breadcrumb: List[Tuple[int, str]] = []
        module_name = None  # Track the top-level module name

        def breadcrumb_text():
            if not breadcrumb:
                return ""
            tail = [t for _, t in breadcrumb][-self.breadcrumb_max_depth:]
            return " > ".join(tail)

        def flush():
            nonlocal curr, curr_tokens
            if not curr:
                return
            pre = breadcrumb_text()
            body = "\n".join(curr).strip()
            # For Lean API docs, include module name if available
            if module_name and not pre:
                pre = module_name
            chunk = f"[Context: {pre}]\n\n{body}" if pre else body
            chunks.append(chunk)
            curr, curr_tokens = [], 0

        def toklen(s: str) -> int:
            return len(self.tok(s, add_special_tokens=False).input_ids)

        def adaptive_overlap(boundary_strength: float) -> int:
            # boundary_strength in [0,1]; higher = stronger boundary -> smaller overlap
            lo, hi = int(self.size * 0.10), min(int(self.size * 0.25), self.base_overlap * 2)
            return int(hi - (hi - lo) * boundary_strength)

        for blk in blocks:
            if blk.type == 'heading':
                # Check if this is a top-level module heading (e.g., "# Module.Name")
                if blk.level == 1 and module_name is None:
                    module_name = blk.text
                
                breadcrumb = [(lvl, txt) for (lvl, txt) in breadcrumb if lvl < blk.level]
                breadcrumb.append((blk.level, blk.text))
                continue

            # Build units - special handling for Lean definitions
            if blk.type == 'lean_def':
                # Treat entire Lean definition as a single atomic unit
                units = [blk.text]
                # Add definition name to metadata if available
                if blk.metadata.get('name'):
                    # Optionally prefix the unit with the name for better context
                    units = [f"## {blk.metadata['name']}\n\n{blk.text}"]
            elif blk.type == 'paragraph':
                units = self._sentences(blk.text)
            elif blk.type == 'list':
                # keep list items intact, merge wrapped lines
                items: List[str] = []
                pending: List[str] = []
                for ln in blk.text.split('\n'):
                    if LIST_BULLET.match(ln) or LIST_NUM.match(ln) or CHECKBOX.match(ln):
                        if pending:
                            items.append(" ".join(s.strip() for s in pending)); pending = []
                        items.append(ln.strip())
                    else:
                        pending.append(ln)
                if pending:
                    items.append(" ".join(s.strip() for s in pending))
                units = items
            elif blk.type in ('code', 'table'):
                units = [blk.text]
            else:
                units = [blk.text]

            # Attempt to add units with similarity-aware boundary decision
            k = 0
            while k < len(units):
                u = units[k]
                ul = toklen(u)
                if curr_tokens + ul <= self.size:
                    curr.append(u); curr_tokens += ul; k += 1; continue
                # Near/over limit: measure similarity valley if we added u
                left_context = curr[-3:]  # last few units already in chunk
                right_context = units[k:k+3]  # next few units
                sim = self.sim_gate.score(left_context, right_context)
                # boundary strength is inverse similarity
                strength = max(0.0, min(1.0, 1.0 - sim))
                # If soft overflow allowed and natural boundary, consider adding then flushing
                if curr and (curr_tokens + ul) <= int(self.size * (1 + SOFT_OVERFLOW_RATIO)) and blk.is_boundary():
                    # Prefer to include and flush at strong boundary (low sim)
                    if strength >= 0.4:  # fairly clear topic break
                        curr.append(u); curr_tokens += ul; flush(); k += 1; continue
                # Otherwise, flush now and start new chunk with u
                flush()
                # If single unit is huge, split by tokens hard
                if ul > self.size:
                    ids = self.tok(u, add_special_tokens=False).input_ids
                    for i in range(0, len(ids), self.size):
                        sub_ids = ids[i:i+self.size]
                        sub = self.tok.decode(sub_ids, skip_special_tokens=True).strip()
                        chunks.append(sub)
                    k += 1
                else:
                    curr = [u]; curr_tokens = ul; k += 1
            # If chunk is quite full and we're at a natural boundary, flush
            # Special case: always flush after a lean_def to keep definitions separate
            if (curr_tokens >= int(self.size * 0.92) and blk.is_boundary()) or blk.type == 'lean_def':
                flush()

        if curr:
            flush()

        # Apply adaptive sentence overlap
        if len(chunks) <= 1:
            return chunks
        overlapped: List[str] = []
        prev_body_sents: List[str] = []
        prev_tail_text = ""
        for idx, ch in enumerate(chunks):
            pre = ""; body = ch
            if ch.startswith("[Context:"):
                parts = ch.split("\n\n", 1)
                if len(parts) == 2:
                    pre, body = parts[0], parts[1]
            sents = [s.strip() for s in sent_tokenize(body) if s.strip()]
            if idx == 0:
                overlapped.append(ch)
                prev_body_sents = sents
                prev_tail_text = body[-200:]
                continue
            # boundary strength from prev tail to current head
            sim = self.sim_gate.score(prev_body_sents[-3:], sents[:3])
            strength = max(0.0, min(1.0, 1.0 - sim))
            ov_tokens = adaptive_overlap(strength)
            # collect tail sentences from prev up to ov_tokens
            tail = []
            acc = 0
            for s in reversed(prev_body_sents):
                tl = toklen(s)
                if acc + tl <= ov_tokens:
                    tail.insert(0, s); acc += tl
                else:
                    break
            combined = ((" ".join(tail) + " ") if tail else "") + body
            overlapped.append((pre + "\n\n" if pre else "") + combined)
            prev_body_sents = sents
            prev_tail_text = body[-200:]
        return overlapped


# ---------------------------
# Pipeline
# ---------------------------

class DataPrep:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, min_words: int = DEFAULT_MIN_WORDS):
        self.output_dir = output_dir
        if chunk_overlap / max(chunk_size, 1) > MAX_ALLOWED_OVERLAP_RATIO:
            logger.warning(f"Requested overlap {chunk_overlap} exceeds 25% of chunk size; clamping to {int(chunk_size * MAX_ALLOWED_OVERLAP_RATIO)}")
            chunk_overlap = int(chunk_size * MAX_ALLOWED_OVERLAP_RATIO)
        if not (0.10 <= (chunk_overlap / max(chunk_size,1)) <= 0.20):
            logger.info("Overlap ratio outside recommended 10–20% window (may increase cost or reduce recall).")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_words = min_words
        # Standard processors
        self.processors = {
            '.pdf': PDFProcessor(),
            '.docx': DocxProcessor(),
            '.txt': TextProcessor(),
            '.md': TextProcessor(),
            '.html': HTMLFileProcessor(),
            '.htm': HTMLFileProcessor(),
            'web': WebPageProcessor(),
        }
        # Specialized processor for Lean API docs
        self.lean_processor = LeanMarkdownProcessor()
        os.makedirs(output_dir, exist_ok=True)
        self.document_index = {}
    
    def _is_lean_api_doc(self, path: str) -> bool:
        """Check if this is a Lean API documentation markdown file."""
        if not path.endswith('.md'):
            return False
        # Check if path contains lean-api or mathlib4_docs patterns
        path_lower = path.lower()
        return ('lean-api' in path_lower or 
                'lean_api' in path_lower or 
                'mathlib4_docs' in os.path.basename(path_lower))

    def is_url(self, path: str) -> bool:
        try:
            res = urlparse(path)
            return all([res.scheme, res.netloc])
        except Exception:
            return False

    def get_relative_path(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.output_dir)
        except Exception:
            return path

    def process_source(self, source_path: str) -> Tuple[bool, str]:
        try:
            if self.is_url(source_path):
                text = self.processors['web'].extract_text(source_path)
                if not text:
                    logger.warning(f"No text extracted from {source_path}")
                    return False, ""
                url_parts = urlparse(source_path)
                safe_filename = f"{url_parts.netloc.replace('.', '_')}_{hash(source_path) % 10000:04d}.txt"
                output_path = os.path.join(self.output_dir, safe_filename)
            else:
                file_path = Path(source_path)
                if not file_path.exists():
                    logger.error(f"File not found: {source_path}")
                    return False, ""
                ext = file_path.suffix.lower()
                if ext not in self.processors:
                    logger.warning(f"Unsupported file type: {ext}")
                    return False, ""
                
                # Use specialized processor for Lean API docs
                if self._is_lean_api_doc(str(file_path)):
                    logger.info(f"Using Lean markdown processor for {source_path}")
                    text = self.lean_processor.extract_text(str(file_path))
                else:
                    text = self.processors[ext].extract_text(str(file_path))
                
                if not text:
                    logger.warning(f"No text extracted from {source_path}")
                    return False, ""
                output_path = os.path.join(self.output_dir, f"{file_path.stem}_processed.txt")

            # Clean + normalize + filter
            # Use appropriate processor for cleaning
            if self._is_lean_api_doc(source_path):
                cleaned = self.lean_processor.clean_text(text)
                filtered = self.lean_processor.filter_semantic_content(cleaned)
            else:
                cleaned = self.processors['.txt'].clean_text(text)
                filtered = self.processors['.txt'].filter_semantic_content(cleaned)
            
            if not cleaned:
                logger.warning(f"No text after cleaning for {source_path}")
                return False, ""
            if not filtered:
                logger.warning(f"No text after semantic filtering for {source_path}")
                return False, ""

            # Semantic-aware token chunking with boundary detection
            tokenizer = get_model_tokenizer()
            chunker = SemanticChunker(self.chunk_size, self.chunk_overlap)
            blocks = chunker.blockify(filtered)
            chunks = chunker.pack(blocks)

            # Filter low-content chunks (but allow code/table/lean_def even if short)
            filtered_chunks: List[str] = []
            for ch in chunks:
                # Preserve code blocks, tables, and Lean definitions regardless of word count
                if re.search(r"```", ch) or re.search(r"\n\|.*\|\n", ch) or re.search(r"^source", ch, re.MULTILINE):
                    filtered_chunks.append(ch); continue
                words = re.sub(r'[^\w\s]', ' ', ch).split()
                meaningful = [w for w in words if len(w) > 1 and not w.isdigit()]
                if len(meaningful) >= self.min_words:
                    filtered_chunks.append(ch)

            if not filtered_chunks:
                logger.warning(f"No chunks with sufficient content for {source_path}")
                return False, ""

            # Emit output with per-chunk token counts
            out_parts = []
            for i, ch in enumerate(filtered_chunks, 1):
                tokc = len(tokenizer(ch, add_special_tokens=False).input_ids)
                out_parts.append(f"--- CHUNK {i} TOKENS={tokc} ---\n{ch.strip()}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(out_parts))

            rel_out = self.get_relative_path(output_path)
            rel_src = source_path if self.is_url(source_path) else self.get_relative_path(source_path)
            total_tokens = count_tokens(filtered)
            self.document_index[rel_out] = {
                'source': rel_src,
                'source_type': 'url' if self.is_url(source_path) else 'file',
                'is_lean_api': self._is_lean_api_doc(source_path),
                'size_tokens': total_tokens,
                'size_chars': len(filtered),
                'chunks': len(filtered_chunks),
                'chunk_size_tokens': self.chunk_size,
                'chunk_overlap_tokens': self.chunk_overlap,
                'filtered_out_chunks': len(chunks) - len(filtered_chunks)
            }
            logger.info(f"Processed {source_path} -> {output_path} ({len(filtered_chunks)} chunks)")
            return True, output_path
        except Exception as e:
            logger.error(f"Error processing {source_path}: {str(e)}")
            return False, ""

    def process_sources(self, source_paths: List[str]) -> Dict[str, str]:
        results = {}
        for src in tqdm(source_paths, desc="Processing sources"):
            ok, outp = self.process_source(src)
            if ok:
                results[src] = outp
        self.save_document_index()
        return results

    def save_document_index(self) -> None:
        index_path = os.path.join(self.output_dir, INDEX_FILENAME)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.document_index, f, indent=2)
        logger.info(f"Document index saved to {index_path} (paths relative to output dir)")


def main():
    parser = argparse.ArgumentParser(description="Process documents/web pages for RAG ingestion (semantic, boundary-aware)")
    parser.add_argument("sources", nargs="*", help="Files or URLs to process")
    parser.add_argument("--input-file", "-i", help="Path to a text file with one file path or URL per line")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR, help=f"Directory to save processed files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--chunk-size", "-c", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Maximum tokens per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", "-v", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Token overlap between chunks (recommend 10-20 pct; default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--min-words", "-m", type=int, default=DEFAULT_MIN_WORDS, help=f"Minimum meaningful words per chunk (default: {DEFAULT_MIN_WORDS})")
    args = parser.parse_args()

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    sources = []
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        p = line.strip()
                        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                            p = p[1:-1]
                        sources.append(p)
            logger.info(f"Loaded {len(sources)} sources from {args.input_file}")
        except Exception as e:
            logger.error(f"Error reading input file {args.input_file}: {e}")
            return
    elif args.sources:
        sources = args.sources
    
    if not sources:
        logger.error("No sources provided. Use positional arguments or --input-file.")
        parser.print_help()
        return

    dp = DataPrep(output_dir=args.output_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, min_words=args.min_words)
    results = dp.process_sources(sources)

    print(f"\nProcessed {len(results)} out of {len(sources)} sources")
    print(f"Results saved to: {args.output_dir}")
    print(f"Document index: {os.path.join(args.output_dir, INDEX_FILENAME)}")
    print(f"Chunk size (tokens): {args.chunk_size}, Overlap (tokens): {args.chunk_overlap}, Min words: {args.min_words}")

if __name__ == "__main__":
    main()