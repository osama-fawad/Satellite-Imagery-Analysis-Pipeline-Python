import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

def black_scholes(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str ="call"
        ):
    """
        Black-Scholes Option Pricing Model

        Parameters:
        S : float - Current price of the commodity
        K : float - Strike price of the option
        T : float - Time to maturity in years
        r : float - Risk-free interest rate
        sigma : float - Volatility of the commodity
        option_type : str - "call" or "put"

        Returns:
        float - Theoretical price of the option
    """
    try:
        logger.info("Calculating Black-Scholes option price")

        d1 = ( np.log(S/K) + (r+(sigma**2/2))*T )/ (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T)*norm.cdf(d2)
        elif option_type == "put":
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid price for for option_type. Allowed : [ call , put ]")

        logger.info(f"Calculated price for {option_type.upper()} option: {price:.2f}")
        return price

    except Exception as e:
        logger.error(f"Error in Black-Scholes model: {e}")
        raise

