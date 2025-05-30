import asyncio
import os
import re
import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

# Sitemap URL
SITEMAP_URL = "https://www.valleyair.org/sitemap.xml"

# Output directory
OUTPUT_DIR = "output"

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename."""
    if not name:
        return ""
    name = name.strip().lower()
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = re.sub(r'[^\w\-_\.]', '', name)  # Remove non-alphanumeric characters
    return name[:100]  # Limit filename length

async def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Fetching sitemap from: {SITEMAP_URL}")
    try:
        resp = requests.get(SITEMAP_URL, timeout=30)
        resp.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(resp.content, "xml") # Use 'xml' parser for sitemaps
        urls = [loc.text for loc in soup.find_all("loc")]
        if not urls:
            print("No URLs found in the sitemap.")
            return
        print(f"Found {len(urls)} URLs in the sitemap.")
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return
    except Exception as e:
        print(f"Error parsing sitemap: {e}")
        return

    # Configure markdown generator with content pruning
    prune_filter = PruningContentFilter(
        threshold=0.5, # As in example
        threshold_type="dynamic", # As in example
        min_word_threshold=10 # As in example
    )
    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    # Configure crawler run
    # Using settings from your example web_crawl.py where applicable
    config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header"], # Key requirement
        stream=True,
        verbose=True, # Good for seeing progress
        word_count_threshold=10, # From example
        page_timeout=120000 # 2 minutes, from example
    )

    print("Starting crawl process...")
    async with AsyncWebCrawler() as crawler:
        agen = await crawler.arun_many(urls, config=config)
        
        processed_count = 0
        idx = 0 # Manual index
        async for result in agen:
            page_url = getattr(result, 'url', f'Unknown_URL_at_index_{idx}')

            # Check for errors first
            if getattr(result, 'error', None):
                print(f"\nError processing URL {page_url}: {result.error}")
                idx += 1 # Increment manual index
                continue # Skip to the next result

            page_title = getattr(result, 'title', '') # Safely get title
            markdown_obj = getattr(result, 'markdown', None) # Safely get markdown object
            
            print(f"\nProcessing URL ({idx+1}/{len(urls)}): {page_url}")

            if markdown_obj and getattr(markdown_obj, 'fit_markdown', None):
                fit_markdown = markdown_obj.fit_markdown
                
                # --- Refined Filename Generation Logic ---
                filename_base = "" 

                # 1. Try to use the page title
                if page_title:
                    filename_base = sanitize_filename(page_title)

                # 2. If title didn't yield a usable name, try parts of the URL
                if not filename_base:
                    url_path_segments = [segment for segment in page_url.split('/') if segment]
                    
                    # Remove http/https if they are the first part
                    if url_path_segments and url_path_segments[0].lower() in ['http:', 'https:']:
                        url_path_segments.pop(0)
                    # Remove domain part if it's the only thing left or to get to actual path segments
                    if url_path_segments and re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", url_path_segments[0]):
                         url_path_segments.pop(0) # Remove domain to focus on path

                    if url_path_segments: # If there are path segments left
                        candidate_from_url = url_path_segments[-1] # Use the last path segment
                        # Remove common web extensions from the segment if present
                        if '.' in candidate_from_url:
                            name_part, ext_part = candidate_from_url.rsplit('.', 1)
                            if ext_part.lower() in ['html', 'htm', 'php', 'asp', 'aspx', 'xml', 'org', 'com', 'net']: # Added common TLDs that might be path endings
                                candidate_from_url = name_part
                        
                        filename_base = sanitize_filename(candidate_from_url)
                    
                    # If path segments didn't yield a name (e.g. root URL, or path sanitized to empty)
                    # try the domain itself as a last resort from URL.
                    if not filename_base:
                        domain_match = re.search(r"https?://([^/]+)", page_url)
                        if domain_match:
                            domain_name = domain_match.group(1)
                            filename_base = sanitize_filename(domain_name)

                # 3. If all above attempts fail, use a generic indexed name
                if not filename_base:
                    filename_base = f"untitled_page_{idx+1}"
                # --- End of Refined Filename Generation Logic ---

                filename = os.path.join(OUTPUT_DIR, f"{filename_base}.md")
                
                # Ensure filename is unique
                counter = 1
                temp_filename = filename
                while os.path.exists(temp_filename):
                    temp_filename = os.path.join(OUTPUT_DIR, f"{filename_base}_{counter}.md")
                    counter += 1
                filename = temp_filename

                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"{page_url}\n\n") # First line is the full URL
                        f.write(fit_markdown)
                    print(f"Successfully wrote content to {filename}")
                    processed_count += 1
                except IOError as e:
                    print(f"Error writing file {filename}: {e}")
            else:
                status = "No processable markdown"
                # No need to check result.error here as it's handled above
                if not markdown_obj:
                    status = "Markdown object is None"
                elif not getattr(markdown_obj, 'fit_markdown', None):
                    status = "fit_markdown is empty or None"
                print(f"No processable markdown for {page_url}. Status: {status}")
                if markdown_obj and getattr(markdown_obj, 'raw_markdown', None):
                    print(f"  Raw markdown was available but fit_markdown was not.")
            idx += 1 # Increment manual index


    print(f"\nCrawling process finished. Processed {processed_count} out of {len(urls)} URLs with content.")

if __name__ == "__main__":
    asyncio.run(main())
