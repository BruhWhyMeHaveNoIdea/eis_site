import argparse
from spiders.links_spider import LinkSpider
from spiders.tenders_spider import TenderSpider


def parse_args():
    parser = argparse.ArgumentParser(
        description='Парсинг тендеров с zakupki.gov.ru'
    )
    parser.add_argument('start_page', type=int, help='Начальная страница пагинации')
    parser.add_argument('end_page', type=int, help='Конечная страница пагинации')
    parser.add_argument('per_page', type=int, help='Количество записей на странице')
    parser.add_argument('pub_date', type=str, help='Дата публикации в формате DD.MM.YYYY')
    parser.add_argument('--close-date', type=str, default=None, help='Дата закрытия в формате DD.MM.YYYY (опционально)')
    return parser.parse_args()


def main():
    args = parse_args()
    close_date = args.close_date if args.close_date else ""

    # Сначала парсим ссылки
    all_links = LinkSpider(
        start_page=args.start_page,
        end_page=args.end_page,
        per_page=args.per_page,
        pub_date=args.pub_date,
        close_date=close_date
    ).start()
    all_links.items.to_json('art/links.json', indent=True)

    # Парсим их содержимое
    result = TenderSpider(
        start_page=args.start_page,
        end_page=args.end_page,
        per_page=args.per_page,
        pub_date=args.pub_date,
        close_date=close_date
    ).start()
    result.items.to_json('art/tenders.json', indent=True)


if __name__ == "__main__":
    main()
