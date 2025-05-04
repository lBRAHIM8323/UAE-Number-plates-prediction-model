import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

# Setup headless browser
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-notifications")
options.add_argument("--disable-extensions")

# Create output directory if it doesn't exist
output_dir = "scraped_html"
os.makedirs(output_dir, exist_ok=True)

def main():
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        start_page = 3
        end_page = 103 #2462
        pages_per_file = 100 #500
        
        for start_chunk in range(start_page, end_page + 1, pages_per_file):
            end_chunk = min(start_chunk + pages_per_file - 1, end_page)
            
            logging.info(f"Starting chunk from page {start_chunk} to {end_chunk}")
            
            combined_html = "<!DOCTYPE html>\n<html>\n<head>\n"
            combined_html += "<meta charset='utf-8'>\n"
            combined_html += f"<title>Combined HTML: Pages {start_chunk}-{end_chunk}</title>\n"
            combined_html += "<style>\n"
            combined_html += ".page-container { border: 1px solid #ccc; margin: 20px 0; padding: 10px; }\n"
            combined_html += ".page-header { background: #f0f0f0; padding: 5px; margin-bottom: 10px; }\n"
            combined_html += "</style>\n"
            combined_html += "</head>\n<body>\n"
            combined_html += f"<h1>Combined pages from {start_chunk} to {end_chunk}</h1>\n"
            
            pages_added = 0
            
            for page in range(start_chunk, end_chunk + 1):
                url = f"https://xplate.com/en/numbers/license-plates?page={page}"
                logging.info(f"Scraping page {page}...")
                
                try:
                    driver.get(url)
                    
                    # Wait for content to load - adjust selector based on the actual page structure
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".number-card, body"))
                        )
                    except Exception as wait_error:
                        logging.warning(f"Wait timeout on page {page}: {wait_error}")
                    
                    # Get page HTML and add to combined HTML
                    page_html = driver.page_source
                    combined_html += f"<div class='page-container'>\n"
                    combined_html += f"<div class='page-header'><h2>Page {page}</h2></div>\n"
                    combined_html += f"<div class='page-content'>{page_html}</div>\n"
                    combined_html += "</div>\n"
                    
                    pages_added += 1
                    logging.info(f"Successfully added page {page}")
                    
                    # Add a small delay to avoid overloading the server
                    time.sleep(1)
                    
                except Exception as e:
                    logging.error(f"Error on page {page}: {e}")
                    # Add error notification to the HTML
                    combined_html += f"<div class='page-container error'>\n"
                    combined_html += f"<div class='page-header'><h2>Page {page} - ERROR</h2></div>\n"
                    combined_html += f"<div class='page-content'>Error: {str(e)}</div>\n"
                    combined_html += "</div>\n"
            
            # Close the HTML document
            combined_html += "</body>\n</html>"
            
            # Save the combined HTML to a file
            filename = os.path.join(output_dir, f"scrapped_{start_chunk}_{end_chunk}.html")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(combined_html)
            
            logging.info(f"Successfully saved {pages_added} pages to {filename}")
            
        driver.quit()
        logging.info("Scraping completed successfully")
        
    except Exception as e:
        logging.critical(f"Critical error in main process: {e}")
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    main()