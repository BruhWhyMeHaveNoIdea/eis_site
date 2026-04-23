import json
from cfg.validate_config import validate_params


class LinksCollector():
    '''Класс для объектов, содержащих все ссылки на тендеры в соответствии с фильтрами'''

    def __init__(self, start_page: int, end_page: int, per_page: int, pub_date: str, close_date: str = ""):
        # Валидируем параметры
        self.config = validate_params(start_page, end_page, per_page, pub_date, close_date)
        
        self.record_per_page = self.config["pagination"]["per_page"]
        self.start_page = self.config["pagination"]["start_page"]
        self.end_page = self.config["pagination"]["end_page"]
        self.pub_date = self.config["filters"]["pub_date"]
        self.close_date = self.config["filters"]["close_date"]
        
        self.pagination_links = []
        self.tenders_links = []

    def collect_pagination(self):
        '''Формируем список страниц с учетом пагинации'''

        self.pagination_links = [
        (
            f"https://zakupki.gov.ru/epz/order/extendedsearch/results.html"
            f"?morphology=on"
            f"&pageNumber={page_num}"
            f"&sortDirection=false"
            f"&recordsPerPage=_{self.record_per_page}"
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
    
    def collect_tenders_links(self):
        '''Формируем список ссылок на тендеры. Берем из art/links.json'''

        with open('art/links.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.tenders_links = [
            'https://zakupki.gov.ru/' + item['link'] for item in data
        ]
        return self.tenders_links