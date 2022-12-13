import json
import requests
from bs4 import BeautifulSoup
import os
import threading

base_url = "https://dermnetnz.org"


def main():
    
    with open("dermnetnz.json", "r") as file:
        data = json.load(file)
            
    threads = []
    
    
    for item in data:
        formatted_name = format_name(item["name"])
        thread = threading.Thread(target=download_images, args=(base_url+item["url"], formatted_name))
        threads.append(thread)
        
        
        
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()



def format_name(name):
    return name.replace(" images", "").lower()



def download_images(url, label):
    
    dir = f"data/{label}/".replace(" ", "_").replace("-", "_").replace(",", "")
    
    if (len(os.listdir(dir)) > 0):
        return
    
    if os.path.exists(dir):
        os.rmdir(dir)

    os.mkdir(dir)

    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    images = soup.find_all('div', {"class" : "imageLinkBlock__item__image"})

    for img in images:
        src = img.find("img").get("src")

        response = requests.get(base_url+src)
        filename = src.split("/")[-1].replace(" ", "_").replace("-", "_")

        with open(dir+filename, "wb") as file:
            file.write(response.content)




if __name__ == "__main__":
    main()