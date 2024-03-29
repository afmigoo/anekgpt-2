import requests
from time import sleep
from pprint import pprint
from tqdm import tqdm

from bs4 import BeautifulSoup

save_to = 'anek_data.txt'

PAGE = 'https://nekdo.ru/short/{page}/'

def get_page(page: int, max_tries: int = 10) -> str:
    if max_tries < 0: 
        return None

    try:
        resp = requests.get(PAGE.format(page=page), timeout=10)
    except requests.Timeout:
        print(f"Timed out")
        sleep(10)
        return get_page(page=page, max_tries=max_tries-1)
    if not resp.ok:
        print(f"Resp not ok: {resp}")
        sleep(10)
        return get_page(page=page, max_tries=max_tries-1)
    
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
    with open(save_to, 'a', encoding='utf-8') as f:
        f.write('\n\n'.join(aneks))

def main():
    SPART_PAGE = 1186
    TOTAL_PAGES = 1805
    total_aneks = 0

    print(f'Overwrite {save_to}?')
    ans = input()
    if ans == 'y':
        with open(save_to, 'w') as f:
            f.write('')

    pb = tqdm(list(range(SPART_PAGE, TOTAL_PAGES + 1)),
              initial=SPART_PAGE,
              total=TOTAL_PAGES)
    for i in pb:
        html = get_page(i)
        aneks = parse_aneks(html)
        write_aneks(aneks)
        
        total_aneks += len(aneks)
        pb.set_description_str(f"total: {total_aneks}")
        sleep(2)

if __name__ == '__main__':
    main()
