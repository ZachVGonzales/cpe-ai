#!/usr/bin/env python3
"""
Web scraper that crawls a website and its subpages up to a specified depth.
Saves all scraped pages as text files in a specified directory.
"""

import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
import hashlib
from typing import Set, Dict
    
import html
from bs4 import NavigableString, Tag

LEAN_LANG_ALIASES = {"lean", "language-lean", "lean4", "language-lean4"}


class WebScraper:
    def __init__(self, base_url: str, output_dir: str, max_depth: int = 2, delay: float = 1.0):
        """
        Initialize the web scraper.
        
        Args:
            base_url: The starting URL to scrape
            output_dir: Directory to save scraped pages
            max_depth: Maximum depth of subpages to crawl (0 = only base page)
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse base domain to only crawl same-domain links
        self.base_domain = urlparse(base_url).netloc
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        parsed = urlparse(url)
        return (
            parsed.scheme in ['http', 'https'] and
            parsed.netloc == self.base_domain
        )
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments (keep query and trailing slash)."""
        parsed = urlparse(url)
        # keep trailing slash exactly as-is; just drop the fragment
        path = parsed.path or '/'
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized

    def _guess_language(tag: Tag) -> str | None:
        """Try to infer language from class names on <code>/<pre>."""
        classes = set(tag.get("class") or [])
        # Common patterns: language-lean, lang-lean, lean
        for c in classes:
            c_low = c.lower()
            if c_low in LEAN_LANG_ALIASES:
                return "lean"
            if c_low.startswith(("language-", "lang-")):
                return c_low.split("-", 1)[1]
        # data-lang attributes sometimes appear
        data_lang = (tag.get("data-lang") or "").lower()
        if data_lang in LEAN_LANG_ALIASES:
            return "lean"
        return None

    def _fence(code: str, lang: str | None) -> str:
        fence_lang = lang if lang else ""
        # ensure trailing newline for nice fences
        if not code.endswith("\n"):
            code += "\n"
        return f"```{fence_lang}\n{code}```\n"

    def _extract_structured(self, html_bytes: bytes, base_url: str) -> dict:
        """
        Return a dict with:
        - title: page title
        - text_md: markdown text with fenced code blocks preserved
        - code_snippets: list of {language, code} blocks
        """
        soup = BeautifulSoup(html_bytes, "html.parser")

        # Title
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")

        # Remove obvious boilerplate; keep code elements
        for t in soup.find_all(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            t.decompose()

        # Collect code blocks first (so we can preserve indentation)
        code_snippets = []
        for pre in soup.find_all("pre"):
            code_tag = pre.find("code")
            if code_tag:
                lang = self._guess_language(code_tag)
                code = code_tag.get_text("\n")
            else:
                lang = self._guess_language(pre)
                code = pre.get_text("\n")
            code = html.unescape(code)
            # Normalize tabs → 4 spaces (optional)
            code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
            code_snippets.append({"language": lang, "code": code})
            # Replace the pre block with a fenced placeholder in the DOM to keep flow
            placeholder = soup.new_tag("div")
            placeholder.string = self._fence(code, lang)
            pre.replace_with(placeholder)

        # Convert remaining HTML to simple markdown-ish text
        # Minimal, dependency-free: turn h1/h2/h3 into markdown headers, lists into bullets.
        def _to_md(node: Tag | NavigableString) -> str:
            if isinstance(node, NavigableString):
                return str(node)
            if isinstance(node, Tag):
                name = node.name.lower()
                if name in {"h1", "h2", "h3"}:
                    hashes = {"h1": "#", "h2": "##", "h3": "###"}[name]
                    inner = "".join(_to_md(c) for c in node.children).strip()
                    return f"\n{hashes} {inner}\n\n"
                if name in {"p"}:
                    inner = "".join(_to_md(c) for c in node.children).strip()
                    return f"{inner}\n\n" if inner else ""
                if name in {"ul"}:
                    items = []
                    for li in node.find_all("li", recursive=False):
                        line = "".join(_to_md(c) for c in li.children).strip()
                        if line:
                            items.append(f"- {line}")
                    return "\n".join(items) + ("\n\n" if items else "")
                if name in {"ol"}:
                    items = []
                    idx = 1
                    for li in node.find_all("li", recursive=False):
                        line = "".join(_to_md(c) for c in li.children).strip()
                        if line:
                            items.append(f"{idx}. {line}")
                            idx += 1
                    return "\n".join(items) + ("\n\n" if items else "")
                if name in {"code"}:
                    # Inline code (not already handled by <pre>)
                    inner = node.get_text()
                    return f"`{inner}`"
                if name in {"br"}:
                    return "\n"
                # Default: recurse
                return "".join(_to_md(c) for c in node.children)
            return ""

        body = soup.body or soup  # fallback
        text_md = _to_md(body)

        # Header with metadata
        header = []
        if title:
            header.append(f"# {title}")
        header.append(f"> Source: {base_url}\n")
        text_md = "\n\n".join(header) + "\n" + text_md

        return {"title": title, "text_md": text_md.strip() + "\n", "code_snippets": code_snippets}


    def _scrape_page(self, url: str) -> tuple[dict, Set[str]]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Skip non-HTML
            if "text/html" not in response.headers.get("Content-Type", ""):
                return {"title": "", "text_md": "", "code_snippets": []}, set()

            # Parse once for links
            link_soup = BeautifulSoup(response.content, "html.parser")
            links = self._extract_links(link_soup, url)

            # Structured extraction for markdown + code
            structured = self._extract_structured(response.content, url)
            return structured, links
        except Exception as e:
            print(f"Error scraping {url}: {e}", file=sys.stderr)
            return {"title": "", "text_md": "", "code_snippets": []}, set()



    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> Set[str]:
        links = set()

        # Use the unnormalized current_url for urljoin (preserves slash semantics)
        for a in soup.find_all('a', href=True):
            absolute = urljoin(current_url, a['href'])
            normalized = self._normalize_url(absolute)
            if self._is_valid_url(normalized):
                links.add(normalized)

        for iframe in soup.find_all('iframe', src=True):
            iframe_url = urljoin(current_url, iframe['src'])
            normalized_iframe = self._normalize_url(iframe_url)
            if self._is_valid_url(normalized_iframe):
                links.add(normalized_iframe)
                try:
                    r = self.session.get(normalized_iframe, timeout=10)
                    r.raise_for_status()
                    iframe_soup = BeautifulSoup(r.content, 'html.parser')
                    for a in iframe_soup.find_all('a', href=True):
                        absolute = urljoin(normalized_iframe, a['href'])
                        normalized = self._normalize_url(absolute)
                        if self._is_valid_url(normalized):
                            links.add(normalized)
                except Exception as e:
                    print(f"Warning: Could not fetch iframe {normalized_iframe}: {e}", file=sys.stderr)
        return links

    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename."""
        # Create a hash-based filename to avoid filesystem issues
        url_hash = hashlib.md5(url.encode()).hexdigest()
        parsed = urlparse(url)
        path = parsed.path.strip('/').replace('/', '_') or 'index'
        
        # Limit path length and sanitize
        path = path[:100]
        filename = f"{path}_{url_hash[:8]}.txt"
        return filename
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract readable text content from page."""
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up multiple newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    
    def _save_page(self, url: str, structured: dict):
        """Save page as .md and optional .lean.txt (only the Lean code)."""
        filename_root = Path(self._url_to_filename(url)).with_suffix("")  # strip .txt
        md_path = self.output_dir / f"{filename_root}.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(structured["text_md"])
            print(f"Saved: {url} -> {md_path.name}")
        except Exception as e:
            print(f"Error saving markdown for {url}: {e}", file=sys.stderr)

        # Save Lean-only concatenated code (optional but great for code-aware indexing)
        lean_snips = [c["code"] for c in structured["code_snippets"] if (c["language"] or "") == "lean"]
        if lean_snips:
            lean_code_path = self.output_dir / f"{filename_root}.lean.txt"
            try:
                with open(lean_code_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(lean_snips).rstrip() + "\n")
                print(f"Saved Lean code: {lean_code_path.name}")
            except Exception as e:
                print(f"Error saving Lean code for {url}: {e}", file=sys.stderr)
    
    def crawl(self, url: str = None, depth: int = 0):
        """
        Recursively crawl pages starting from the given URL.
        
        Args:
            url: URL to start crawling from (uses base_url if None)
            depth: Current depth level
        """
        if url is None:
            url = self.base_url
        
        # Normalize URL
        url = self._normalize_url(url)
        
        # Check if already visited or max depth reached
        if url in self.visited_urls or depth > self.max_depth:
            return
        
        # Mark as visited
        self.visited_urls.add(url)
        
        print(f"[Depth {depth}] Scraping: {url}")
        
        # Scrape the page
        content, links = self._scrape_page(url)
        
        if content:
            self._save_page(url, content)
        
        # If we haven't reached max depth, crawl subpages
        if depth < self.max_depth:
            for link in links:
                if link not in self.visited_urls:
                    time.sleep(self.delay)  # Be polite to the server
                    self.crawl(link, depth + 1)
    
    def run(self):
        """Start the crawling process."""
        print(f"Starting web scraper...")
        print(f"Base URL: {self.base_url}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max depth: {self.max_depth}")
        print(f"Delay between requests: {self.delay}s")
        print("-" * 80)
        
        self.crawl()
        
        print("-" * 80)
        print(f"Scraping complete! Visited {len(self.visited_urls)} pages.")
        print(f"Files saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape a website and its subpages up to a specified depth.'
    )
    parser.add_argument(
        'url',
        help='The base URL to start scraping from'
    )
    parser.add_argument(
        '-o', '--output',
        default='./scraped_pages',
        help='Output directory for scraped pages (default: ./scraped_pages)'
    )
    parser.add_argument(
        '-d', '--depth',
        type=int,
        default=2,
        help='Maximum depth of subpages to crawl (default: 2)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Create and run scraper
    scraper = WebScraper(
        base_url=args.url,
        output_dir=args.output,
        max_depth=args.depth,
        delay=args.delay
    )
    
    try:
        scraper.run()
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user.")
        print(f"Visited {len(scraper.visited_urls)} pages before interruption.")
        sys.exit(1)


if __name__ == '__main__':
    main()
