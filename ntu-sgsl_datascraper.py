import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL of the website - containing all the signs on the website
base_url = "https://blogs.ntu.edu.sg/sgslsignbank/signs/"

# Folder for the images/ gifs
image_folder = "Data/ntu_sgsl"
os.makedirs(image_folder, exist_ok = True) # create folder if it does not exist

# Download images or gifs and save it to the folder
def download_signs(image_url, save_name):
    try:
        response = requests.get(image_url, stream = True)
        response.raise_for_status() # raise error for HTTP issues
        file_path = os.path.join(image_folder, save_name)

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"Saved: {save_name}")

    except Exception as e:
        print(f"Failed t0 download {save_name} - {image_url}: {e}")

# Main function - Scrape the website for image or gifs of signs
def sign_scraper():
    try:
        # Request the webpage - which contains the list of signs
        response = requests.get(base_url)
        response.raise_for_status()

        # Parse the webpage content
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all the relevant links of the signs based on the specified class
        links = soup.find_all("a", class_="sign btn btn-red")

        for link in links:
            if 'href' not in link.attrs:
                continue
            href = link['href']
            page_url = href if href.startswith("http") else base_url + href

            # Visit each linked page - shows the sign
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"} # 
            page_response = requests.get(page_url, headers = headers)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, "html.parser")

            # Find the image or gif with the specified class
            img_tag = page_soup.find("img", class_="w-100 img-fluid mb-2")
            if img_tag and 'src' in img_tag.attrs and 'alt' in img_tag.attrs:
                img_src = img_tag['src']
                file_extenstion = os.path.splitext(img_src.split("?")[0])[-1] 
                alt_name = img_tag['alt'].replace(" ", "_").replace("/", "_").replace("-demo", "") + file_extenstion # get the alt name of the image as the label

                # Ensure the image source URL is absolute
                img_url = img_src if img_src.startswith("http") else base_url + img_src

                # Download and save the image
                download_signs(img_url, alt_name)

    except Exception as e:
        print(f"Error while scraping: {e}")

if __name__ == "__main__":
    sign_scraper()