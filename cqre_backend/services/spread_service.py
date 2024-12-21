from database.db_connection import get_db_connection
import logging

logger = logging.getLogger(__name__)

class SpreadService:
    def save_spread_to_db(self, wti_price, brent_price, spread):
        """Store wti price, brent price and spread"""

        try:
            conn = get_db_connection()
            cursor =conn.cursor()
            cursor.execute(
                "INSERT INTO spread_history(wti_price, brent_price, spread) "
                "VALUES (%s, %s, %s)", wti_price, brent_price, spread
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("Spread stored in database")

        except Exception as e:
            logger.error(f"Error storing spread in database: {e}")

    def fetch_spread_history(self, limit=30):
        """Store wti price, brent price and spread"""

        try:
            conn = get_db_connection()
            cursor =conn.cursor()
            cursor.execute(
                "SELECT wti_price, brent_price, spread FROM spread_history ORDER BY timestamp DESC LIMIT %s",
                (limit,)
            )
            row = cursor.fetchall()
            cursor.close()
            conn.close()
            logger.info("Fetched spread history from database")
            return rows
        except Exception as e:
            logger.error(f"Error fetching spread from database: {e}")
            return []
