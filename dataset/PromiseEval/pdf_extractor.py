import json
import requests
import os
from urllib.parse import urlparse


SAVE_DIR = "dataset/PromiseEval/pdfs"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("dataset/PromiseEval/ml_promise.json", "r", encoding='utf-8') as f:
    ml_promise = json.load(f)

urls = list(set([item["URL"] for item in ml_promise]))

failed_urls = []
url_pdf_map = {}

for url in urls:
    try:
        # Get filename from URL or use a default name
        filename = os.path.basename(urlparse(url).path)
        if not filename.endswith('.pdf'):
            filename = f"document_{len(url_pdf_map)}.pdf"
        
        # Download the PDF
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the PDF
        pdf_path = os.path.join(SAVE_DIR, filename)
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Add to mapping if successful
        url_pdf_map[url] = pdf_path
        print(f"Successfully downloaded: {url}")
        
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        failed_urls.append(url)

# Save the mappings and failed URLs for reference
with open(os.path.join(SAVE_DIR, "url_pdf_mapping.json"), "w") as f:
    json.dump(url_pdf_map, f, indent=2)

with open(os.path.join(SAVE_DIR, "failed_urls.json"), "w") as f:
    json.dump(failed_urls, f, indent=2)

print(f"\nDownload Summary:")
print(f"Total URLs: {len(urls)}")
print(f"Successful downloads: {len(url_pdf_map)}")
print(f"Failed downloads: {len(failed_urls)}")




