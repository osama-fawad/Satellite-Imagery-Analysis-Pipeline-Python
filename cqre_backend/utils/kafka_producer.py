from kafka import KafkaProducer
from data_fetcher import PriceFetcher
import json
import time
import logging
import random

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
            # Generate wti and brent price
            wti_price = round(random.uniform(65.0, 95.0), 2)
            brent_price = round(wti_price + random.uniform(1.0, 5.0), 2)

            prices = { "wti_price": wti_price, "brent_price": brent_price }
            #prices = fetcher.load_prices() #Currently commeted out not not reach api data fetching limit

            producer.send(topic, prices)

            if prices:
                print(f"WTI: {prices['wti_price']}, Brent: {prices['brent_price']}")
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Producer stopped.")
    finally:
        producer.close()

if __name__ == "__main__":
    produce_prices()