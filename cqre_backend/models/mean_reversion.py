import numpy as np
import logging

import sys
sys.path.append('../')

logger = logging.getLogger(__name__)

def calculate_spread(wti_price, brent_price):
    """Calculate the price spread between brent and wti crude oil"""

    try:
        spread = brent_price - wti_price
        logger.info(f"Calulated spread: {spread:.4f}")
        return spread
    except Exception as e:
        logger.error(f"Error calculating spread: {e}")
        raise

def calculate_statistics(spread_history):
    """Calculate the mean and the standard deviation of the spread history"""

    try:
        mean = np.mean(spread_history)
        standard_deviation = np.std(spread_history)
        logger.info(f"Calculated mean: {mean:.4f}, standard deviation: {standard_deviation:4f}")
        return mean,standard_deviation

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise

def generate_signal(spread,
                    mean,
                    standard_deviation,
                    upper_threshold=2,
                    lower_threshold=-2):
    """Generate trading signal based on Z-score of the spread"""

    try:
        z_score = (spread-mean)/standard_deviation
        logger.info(f"Z-Score: {z_score:.4f}")

        if z_score > upper_threshold:
            signal = "SELL Brent, BUY WTI"

        elif z_score < lower_threshold:
            signal = "BUY Brent, SELL WTI"
        else:
            signal = "HOLD"

        logger.info(f"Generated Signal: {signal}")

        return signal
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise
