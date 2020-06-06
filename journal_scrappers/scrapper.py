import io
import requests as req
from bs4 import BeautifulSoup

def scrap_swsys():

    results_path = "texts/"

    start_url = "http://swsys.ru/"
    all_articles_address = "http://swsys.ru/index.php?page=all_article&lang="
    resp = req.get(all_articles_address)

    # get all hrefs for articles
    soup = BeautifulSoup(resp.text, 'lxml')
    links = []
    for a in soup.find("div", id="col3_content").find_all("a", href=True):
        links.append(a['href'])
    
    # get last N articles and parse abstract
    articles_count = len(links)
    ind = 0
    print("Articles found: " + str(articles_count))
    while (ind < articles_count):
        cur_article_url = start_url + links[ind]
        cur_art_resp = req.get(cur_article_url)
        ind = ind + 1

        new_soup = BeautifulSoup(cur_art_resp.text, 'lxml')
        all_ems = []
        for em in new_soup.find("div", id="col3_content").find_all("em"):
            all_ems.append(em.text)

        if (all_ems[0] == ""):
            print("Article #" + str(ind) + ": " + new_soup.title.text + " HAS NO ABSTRACT")
        else:
            with io.open(results_path + str(ind) + ".txt", mode='w', encoding='utf-8') as outfile:
                outfile.write(new_soup.title.text + "\n")
                outfile.write(all_ems[0])

def scrap_cloudofscience():
#TODO add title
    results_path = "cloudofscience_texts/"

    start_url = "https://cloudofscience.ru"
    all_articles_address = "https://cloudofscience.ru/publications/archive"
    resp = req.get(all_articles_address, verify=False)

    # get all hrefs for articles
    soup = BeautifulSoup(resp.text, 'lxml')
    links = []
    for a in soup.find_all("div", class_ = "field-item even")[0].find_all("a", href=True):
        links.append(a['href'])

    parts_count = len(links)
    ind = 0
    print("Parts found: " + str(parts_count))
    while (ind < parts_count):
        cur_article_url = start_url + links[ind]
        cur_art_resp = req.get(cur_article_url, verify=False)
        ind = ind + 1

        new_soup = BeautifulSoup(cur_art_resp.text, 'lxml')
        print(ind)
        print(cur_article_url)
        sub_ind = 0
        for el in new_soup.find_all("div", class_ = "annotation"):
            with io.open(results_path + str(ind) +"_" + str(sub_ind) + ".txt", mode='w', encoding='utf-8') as outfile:
                annotation_span = el.find_all("span")
                if (len(annotation_span) > 0):
                    outfile.write(el.find_next("span").text)
                else:
                    outfile.write(el.text[11:])
            sub_ind += 1

def scrap_ius():
    results_path = "ius_texts/"

    start_url = "http://www.i-us.ru/"
    all_articles_address = "http://www.i-us.ru/index.php/ius/issue/archive"
    resp = req.get(all_articles_address)

    # get all hrefs for articles
    soup = BeautifulSoup(resp.text, 'lxml')
    links = []
    for a in soup.find_all("h2", class_ = "post-title"):
        links.append(a.find_all("a", href=True)[0]['href'])

    parts_count = len(links)
    ind = 0
    print("Parts found: " + str(parts_count))
    while (ind < parts_count):
        cur_part_url = links[ind]
        cur_part_resp = req.get(cur_part_url)
        ind = ind + 1

        new_soup = BeautifulSoup(cur_part_resp.text, 'lxml')
        galleys_links = []
        for a in new_soup.find_all("ul", class_ = "galleys_links"):
            galleys_links.append(a.find_all("a", href=True)[0]['href'])

        sub_ind = -1
        for article_link in galleys_links:
            if (sub_ind < 0): #some special case
                sub_ind += 1
                continue # skip all pdf first link
            print(article_link)
            cur_art_resp = req.get(article_link)
            cur_art_soup = BeautifulSoup(cur_art_resp.text, 'lxml')
            result_str = ""
            title_ar = cur_art_soup.find_all("h1", class_="page_title")
            abstract_ar = cur_art_soup.find_all("div", class_="item abstract")
            if (len(title_ar) > 0):
                result_str = title_ar[0].text
            if (len(abstract_ar) > 0):
                result_str += abstract_ar[0].find_next("span").text
            with io.open(results_path + str(ind) +"_" + str(sub_ind) + ".txt", mode='w', encoding='utf-8') as outfile:
                outfile.write(result_str)
            sub_ind += 1

def scrap_itmag():
    results_path = "itmag_texts/"

    start_url = "https://nbpublish.com/itmag/contents_"
    end_url = ".html"

    start_year = 2012
    end_year = 2020
    start_volume = 1
    end_volume = 4
    links = []
    for year in range(start_year, end_year):
        for vol in range(start_volume, end_volume):
            links.append(start_url + str(year) + "_" + str(vol) + end_url)

    parts_count = len(links)
    ind = 0
    print("Parts found: " + str(parts_count))
    while (ind < parts_count):
        cur_vol_url = links[ind]
        cur_vol_resp = req.get(cur_vol_url, verify=False)
        ind = ind + 1

        print(ind)
        print(cur_vol_url)
        soup = BeautifulSoup(cur_vol_resp.text, 'lxml')
        sub_ind = 0
        add_info_divs = soup.find_all("div", class_="add_info")
        if (len(add_info_divs) == 0):
            continue

        sub_ind = 0
        for el in add_info_divs:
            with io.open(results_path + str(ind) +"_" + str(sub_ind) + ".txt", mode='w', encoding='utf-8') as outfile:
                outfile.write(el.find_next("div").text[11:])
            sub_ind += 1

#scrap_swsys()
#scrap_cloudofscience()
#scrap_ius()
scrap_itmag()