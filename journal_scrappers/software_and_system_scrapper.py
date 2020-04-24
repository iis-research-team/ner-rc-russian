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

def scrap_other():
    print("TODO")

#scrap_swsys()
scrap_other()