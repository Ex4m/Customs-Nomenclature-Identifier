"""
user input ( plastic tableware)

using patterns compare data and return from dictionary most probable (let´s say 5) HS codes

"""
import pandas as pd
import requests as rq
from bs4 import BeautifulSoup as bfs
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import json
import random
import re
import keyboard as key
possible_response = ["y","yes","yap","yeah",","]



def get_results(search_input):
    url = "https://www.zauba.com/shipment_search"
    browser = webdriver.Chrome()
    browser.get(url)
    search_box = browser.find_element(By.ID, value="search") #search je zde unikátní element přímo z požadované stránky
    search_box.send_keys(search_input)  # napíše input do vyhledávače
    search_box.send_keys(Keys.RETURN)   # zmáčkne enter a vyhledá tedy co je třeba
    time.sleep(1)

    response = browser.current_url
    #soup = bfs(response.content,'html.parser')
    try:
        scraper = pd.read_html(rq.get(response).text, attrs={"class": "result-table"})
    except Exception as chyba:
        print(chyba)
        scraper = []
    print(scraper)
    browser.quit()
    return scraper

def random_words(loaded_data):
    element = loaded_data[loaded_data.columns[1]].to_list()
    random_element = random.choice(element)
    words = re.findall(r'\b[a-zA-Z]+\b',random_element)
    words = random.choice(words)
    return words

    """word_list = random_element.split()
    random_word = random.choice(word_list)
    return random_word"""

while True:
    loaded_data= pd.read_json("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", lines = True)
    loaded_data= pd.DataFrame(loaded_data)
    ran_w1 = random_words(loaded_data)
    ran_w2 = random_words(loaded_data)
    two_words = ran_w1 + " " + ran_w2
    

    #search = input("Search HS code using keywords: ")
    scraper = get_results(two_words)
    if scraper == []:
        continue

    scraper = pd.DataFrame(scraper[0])
    print("------------------------------------------------------------------------------")

    column = scraper.columns.to_list()
    print(column)

    print("------------------------------------------------------------------------------")
    #scraper.drop(labels = ["Date","Origin Country","Port of Discharge","Unit","Quantity","Value (INR)","Per Unit (INR)"], axis=1)
    scraper.drop(scraper.columns[[0,3,4,5,6,7,8]], axis=1, inplace=True)
    print(scraper)
    #scraper.to_excel("data_nom.xlsx", index=False, sheet_name='Sheet1', header=True)
    #scraper.to_csv("data_nomenc_csv.csv", index = False)

    #scraper.to_json("Nom_output2", orient = "table")
    file_exist = False
    try:
        #with open ("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", "r") as file:
        #    nom_output_data = json.load(file)
        nom_output_loaded = pd.read_json("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", lines = True)
        file_exist = True
    except FileNotFoundError as chyba:
        print("Error: JSON file not found")
        print(chyba)
        file_exist = False
    except json.decoder.JSONDecodeError as chyba:
        # the file is empty or does not contain valid JSON
        print("Error: JSON file is empty or invalid")
        print(chyba)
        file_exist = False
    except Exception as chyba:
        print(chyba)
        pass
    else:
        # the file was found and the data was loaded successfully
        pass


    if file_exist == False:
        #loaded_df = pd.read_json("Nom_output")
        #scraper.to_excel("Nom_output_excel", sheet_name= "sheet1")
        scraper.to_json("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", orient = "records", lines= True)
        print("new file saved")
        
    else:
        new_set = nom_output_loaded.append(scraper)
        new_set = new_set.dropna()
        new_set = new_set.drop_duplicates()
        new_set = new_set.sort_values("HS Code")
        new_set.to_json("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", orient = "records", lines= True)

        #with open("C:/Users/Exa/Desktop/Python/Customs Nomenclature finder/Nom_output", "a") as original_file:
        #    scraper.to_json(original_file, orient = "records", lines= True)
        #nom_output_data.append(scraper)
        print("new info appended")

    if key.is_pressed("ctrl+9"):
        break
    #response = input("Do you want to run the script again? y/n: ")
    #if response.lower() not in possible_response:
    #    break

    """search = input("Search HS code using keywords: ")

    response = rq.get(f'https://www.zauba.com/import-plastic-table-hs-code.html')

    soup = bfs(response.content,'html.parser')

    scraper = pd.read_html(soup)

    scraper"""



    """response = rq.get(f'https://www.zauba.com/hs_code_search?q={search}')





    <input type="text" id="search" name="search" placeholder="Enter Product description" required="">
    document.querySelector("#search")

    <input type="submit" value="Search" id="submit">"""


