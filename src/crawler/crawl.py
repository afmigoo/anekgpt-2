import requests
from time import sleep
from pprint import pprint
from tqdm import tqdm

from bs4 import BeautifulSoup

#from src.anek_gpt.config import raw_data

TMP = 'data.txt'
PAGE = 'https://nekdo.ru/short/{page}/'

def get_page(page: int, max_tries: int = 10) -> str:
    if max_tries < 0: 
        return None

    resp = requests.get(PAGE.format(page=page), timeout=10)
    if not resp.ok:
        print(f"Resp not ok: {resp}")
        sleep(10)
        get_page(page=page, max_tries=max_tries-1)
    
    html = resp.text
    return html

def parse_aneks(html: str) -> list[str]:
    bs = BeautifulSoup(html, features="html.parser")
    for br in bs.find_all("br"):
        br.replace_with("\n")
    elems = bs.find_all('div', class_='text')
    elems = [e for e in elems]
    aneks = [e.get_text() for e in elems]
    return aneks

def write_aneks(aneks: list[str]):
    with open(TMP, 'a', encoding='utf-8') as f:
        f.write('\n\n'.join(aneks))

def main():
    TOTAL_PAGES = 1805
    total_aneks = 0

    print(f'Overwrite {TMP}?')
    ans = input()
    if ans == 'y':
        with open(TMP, 'w') as f:
            f.write('')

    pb = tqdm(range(1, TOTAL_PAGES + 1))
    for i in pb:
        html = get_page(i)
        aneks = parse_aneks(html)
        write_aneks(aneks)
        
        total_aneks += len(aneks)
        pb.set_description_str(f"total: {total_aneks}")
        sleep(2)

if __name__ == '__main__':
    main()
