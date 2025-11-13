from cpe_ai.services.web_scrapper_service import WebScraper
import sys

def main():
    scraper = WebScraper(
        base_url="https://leanprover-community.github.io/mathlib4_docs/Mathlib.html",
        output_dir="data/vector-store/raw-docs/mathlib_only_docs",
        max_depth=5,
        delay=0.1,
    )

    try:
        scraper.run()
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user.")
        print(f"Visited {len(scraper.visited_urls)} pages before interruption.")
        sys.exit(1)

if __name__ == '__main__':
    main()