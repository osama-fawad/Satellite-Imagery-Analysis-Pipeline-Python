import requests
import logging
import time

logger = logging.getLogger(__name__)

class PriceFetcher:
    ALPHA_VANTAGE_API_KEY = "ORZFQ5AW5CT81HZH"
    WTI_API_URL = "https://www.alphavantage.co/query?function=WTI&interval=1min&apikey="
    BRENT_API_URL = "https://www.alphavantage.co/query?function=BRENT&interval=1min&apikey="

    def fetch_price(self, commodity, api_url):
        """Fetch real time commodity price"""
        try:
            response = requests.get(api_url + self.ALPHA_VANTAGE_API_KEY)
            data = response.json()
            price = float(list[data["data"].values()][0]["value"])
            logger.info(f"{commodity}-price: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fething {commodity} price: {e}")
            return None

    def load_prices(self):
        """Fetch both wti and brent price"""

        wti_price = self.fetch_price("WTI", self.WTI_API_URL)
        brent_price = self.fetch_price("Brent", self.BRENT_API_URL)

        if wti_price and brent_price:
            return {
                "wti_price": wti_price,
                "brent_price": brent_price
            }
        else:
            logger.warning("Some price is not correclty fetched")
            return None

if __name__ == "__main__":
    fetcher = PriceFetcher()
    while True:
        prices = fetcher.load_prices()
        if prices:
            print(f"WTI: {prices['wti_price']}, Brent: {prices['brent_price']}")
        time.sleep(15)
