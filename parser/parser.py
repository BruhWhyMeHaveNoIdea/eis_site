import logging
import sys
import io
import os

# Полностью отключаем логирование ДО импорта scrapling
logging.disable(logging.CRITICAL)
logging.basicConfig = lambda *args, **kwargs: None  # Блокируем basicConfig

from cfg.validate_pars import validate_parser_params
from spiders.links_spider import LinkSpider
from spiders.tenders_spider import TenderSpider
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Parser():
    '''Основной класс парсера, передаем аргументы как атрибуты объекта'''

    pagination_links = []   # Все ссылки с учетом пагинации (переинициализуруем в collect_pagination())
    links = []  # Ссылки на тендеры (переинициализуруем в get_links())
    tenders = []    # Данные по тендерам (переинициализуруем в get_tenders())

    def __init__(self, start_page, end_page, per_page, pub_date, close_date=None):

        validate_parser_params(start_page, end_page, per_page, pub_date, close_date)
        self.start_page = start_page
        self.end_page = end_page
        self.per_page = per_page
        self.pub_date = pub_date
        self.close_date = close_date

    def collect_pagination(self):
        '''Формируем список страниц с учетом пагинации'''

        self.pagination_links = [
        (
            f"https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            f"?morphology=on"
            f"&pageNumber={page_num}"
            f"&sortDirection=false"
            f"&recordsPerPage=_{self.per_page}"
            f"&showLotsInfoHidden=false"
            f"&sortBy=UPDATE_DATE"
            f"&fz44=on&af=on&ca=on"
            f"&currencyIdGeneral=-1"
            f"&publishDateFrom={self.pub_date}"
            f"&applSubmissionCloseDateFrom={self.close_date}"
        )

        for page_num in range(
            self.start_page,
            self.end_page + 1,
        )
    ]
        return self.pagination_links

    def get_links(self):
        '''Собираем все ссылки на тендеры через паука, сохраняем в self.links'''
        link_spider = LinkSpider(
            start_page=self.start_page,
            end_page=self.end_page,
            per_page=self.per_page,
            pub_date=self.pub_date,
            close_date=self.close_date or ""
        ).start()
        full_links = ['https://zakupki.gov.ru/' + item['link'] for item in link_spider.items]
        self.links = full_links

    def get_tenders(self):
        '''Парсим тендеры с помощью паука, сохраняем в self.tenders'''
        tender_spider = TenderSpider(start_urls=self.links).start()
        self.tenders = tender_spider.items

    def start(self):
        '''Упрощенная функция для старта'''
        self.collect_pagination()
        self.get_links()
        self.get_tenders()

        return self.tenders
        
if __name__ == "__main__":
    parser = Parser(1, 10, 5, '10.2.2024')
    result = parser.start()
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)