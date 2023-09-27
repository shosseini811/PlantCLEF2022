import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import os

# # Create a directory to save the images
# os.makedirs('images', exist_ok=True)

def download_image(image_dir):
    # Get the image directory page
    response = requests.get(image_dir)
    response.raise_for_status()  # Check if the request was successful
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Assume the first .jpg link is the image file
    image_link = image_dir + [a['href'] for a in soup.find_all('a') if a['href'].endswith('.jpg')][0]
    image_name = image_link.split('/')[-1]
    
    # Download the image
    response = requests.get(image_link)
    response.raise_for_status()  # Check if the request was successful
    with open(f'data/train/{image_name}', 'wb') as file:
        file.write(response.content)
    print(f'Downloaded image {image_name}')

# Specify the URL
url = 'https://lab.plantnet.org/LifeCLEF/PlantCLEF2022/train/trusted/images/'

# Send an HTTP request to the URL
response = requests.get(url)
response.raise_for_status()  # Check if the request was successful

# Parse the HTML with Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

# Find all image directories
image_dirs = [url + a['href'] for a in soup.find_all('a') if '/' in a['href']]

# Create a directory to save the images
# os.makedirs('images', exist_ok=True)

# Use ThreadPoolExecutor to download the first 10 images concurrently
with ThreadPoolExecutor() as executor:
    executor.map(download_image, image_dirs)
