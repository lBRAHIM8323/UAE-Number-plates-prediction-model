from bs4 import BeautifulSoup
import csv
import re

# Load the HTML content
with open("scraped_html\scrapped_3_102.html", "r", encoding="utf-8") as file:
    html_content = file.read()
    soup = BeautifulSoup(html_content, "lxml")

# Find all license plate card divs (each containing all the information for one plate)
license_plate_cards = soup.find_all("div", class_="number-card")

data = []
url_pattern = re.compile(
    r"https://xplate\.com/en/numbers/license-plates/\d+-(.+?)-code-([a-zA-Z0-9]+)-plate-number-(\d+)"
)

# Process each license plate card
for card in license_plate_cards:
    try:
        # Find the thumb section with the URL
        thumb = card.find("div", class_="thumb")
        if not thumb:
            continue
            
        a_tag = thumb.find("a")
        if not a_tag or "href" not in a_tag.attrs:
            continue
            
        url = a_tag["href"]
        
        # Extract plate details from URL
        match = url_pattern.match(url)
        if not match:
            continue
            
        city, code, plate_number = match.groups()
        
        # Find price within the same card
        price_tag = card.find("span", class_="custom-red dm-white")
        price = price_tag.text.strip() if price_tag else "N/A"
        
        # Find timestamp within the same card
        timestamp_div = card.find("div", class_="d-flex align-items-center meta")
        timestamp = timestamp_div.find("span").text.strip() if timestamp_div and timestamp_div.find("span") else "N/A"
        
        # Append data
        data.append([city, code, plate_number, price, timestamp])
        
    except Exception as e:
        print(f"Error processing a license plate card: {e}")
        continue

# Write to CSV
with open("extracted_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["city", "code", "plate_number", "price", "timestamp"])
    writer.writerows(data)

print(f"✅ Extracted {len(data)} valid rows to extracted_data2.csv")