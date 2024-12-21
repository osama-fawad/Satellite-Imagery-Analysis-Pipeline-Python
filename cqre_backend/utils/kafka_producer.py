from kafka import KafkaProducer
from data_fetcher import PriceFetcher
import json
import time
import logging

logger = logging.getLogger(__name__)

def produce_prices():
    producer = KafkaProducer(
        bootstrap_servers = "localhost:9092",
        value_serializer = lambda v: json.dumps(v).encode("utf-8")
    )
    topic = "commodity-prices"
    fetcher = PriceFetcher()

    try:
        while True:
            prices = { "wti_price": 69.5, "brent_price": 72.9 }
            #prices = fetcher.load_prices() #Currently commeted out not not reach api data fetching limit

            producer.send(topic, prices)

            if prices:
                print(f"WTI: {prices['wti_price']}, Brent: {prices['brent_price']}")
            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Producer stopped.")
    finally:
        producer.close()

if __name__ == "__main__":
    produce_prices()