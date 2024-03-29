from pprint import pprint
from random import choice
from tqdm import tqdm
from time import sleep
from pathlib import Path
import re

import wikipedia

wikipedia.set_lang('ru')
PAGES_COUNT = 50000
SAVE_TO = Path(__file__).parent.joinpath('wiki_data.txt')

def rand_articles(n = 10) -> list[wikipedia.WikipediaPage]:
    articles = []
    while len(articles) < n:
        title = wikipedia.random(min(10, n))
        try:
            articles.append(wikipedia.page(title=title))
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            pass
    return articles

def clear_content(content: str) -> str:
    # removing translated text between round braces (фин. Дом)
    content = re.sub(r'\((?:от )?[а-я]{2,4}\..*\)', ' ', content)
    # removing titles
    content = re.sub(r'(?:=+ ).*(?: =+)', ' ', content)
    # removing all in between quve braces
    content = re.sub(r'\{.*\\.*\}', ' ', content)
    # removing extra linebreaks
    content = re.sub(r'[\s]{2,}', '\n', content)
    return content

def write_article(article: wikipedia.WikipediaPage):
    with open(SAVE_TO, 'a', encoding='utf-8') as f:
        f.write(clear_content(article.content) + "\n\n")

def main():
    pb = tqdm(total=PAGES_COUNT, unit='p')
    pages: list[wikipedia.WikipediaPage] = []
    for _ in range(PAGES_COUNT // 10):
        new_articles = rand_articles(10)
        for article in new_articles:
            write_article(article)
        pb.update(len(new_articles))
        sleep(1)

    pprint(pages)

if __name__ == '__main__':
    main()
