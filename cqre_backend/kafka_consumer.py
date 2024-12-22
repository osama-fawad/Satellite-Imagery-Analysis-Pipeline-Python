from kafka import KafkaConsumer
from models.mean_reversion import calculate_spread, calculate_statistics, generate_signal
from services.spread_service import SpreadService
import json
import time
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SpreadConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            "commodity-prices",
            bootstrap_servers="localhost:9092",
            group_id="trade-group",
            value_deserializer=lambda v: json.loads(v.decode("utf-8"))
        )
        self.spread_service = SpreadService()

    def consume_prices(self):
        logger.info("Start kafka consumer...")

        for message in self.consumer:
            price_data = message.value
            wti_price = price_data.get("wti_price")
            brent_price = price_data.get("brent_price")

            if wti_price and brent_price:
                spread =calculate_spread(wti_price,brent_price)

                self.spread_service.save_spread_to_db(wti_price,brent_price,spread)
                spread_history = self.spread_service.fetch_spread_history()
                spreads = [rows[2] for rows in spread_history]

                mean, standard_deviation = calculate_statistics(spreads)
                signal = generate_signal(spread, mean, standard_deviation)

                logger.info(
                    f"Processed: WTI = {wti_price}, Brent= {brent_price}, Spread={spread:.2f}, Signal={signal}"
                )
            else:
                logger.warning("Invalid price data.")

if __name__ == "__main__":
    logger.info("Starting consumer...")
    consumer = SpreadConsumer()
    consumer.consume_prices()
















