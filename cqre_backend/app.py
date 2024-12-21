from flask import Flask, jsonify, request
import logging
from models.black_scholes import black_scholes
from models.mean_reversion import calculate_spread, calculate_statistics, generate_signal
from services.spread_service import SpreadService

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.route('/option-pricing/black-scholes/', methods = ['POST'])
def option_pricing_black_scholes():
    """
    Endpint for black-scholes model option pricing calculation
    """

    try:
        logger.info("Starting price calculation with black scholes model")
        data = request.json
        S=data.get("S")
        K=data.get("K")
        T=data.get("T")
        r=data.get("r")
        sigma=data.get("sigma")
        option_type=data.get("option_type")

        if not all([S,K,T,r,sigma,option_type]):
            logger.error(f"Some parameters are missing: {e}")
            return jsonify({"Error":"Missing parameters"}), 400

        price = black_scholes(S,K, T, r, sigma, option_type)
        logger.info("Price successfully calculated..")
        return jsonify({"price": price})
    except Exception as e:
        logger.error(f"Error in black scholes option pricing api: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/option-pricing/spread-analysis/', methods = ['POST'])
def spread_analysis():
    """Analyze WTI-Bread spread and generate trading signals"""

    try:
        data = request.json
        wti_price = data.get("wti_price")
        brent_price = data.get("brent_price")

        if not all([wti_price, brent_price]):
            return jsonify({"Error: Missing parameters"})


        """Calculate the spread and update spread history for rolling 30 spreads"""
        spread = calculate_spread(wti_price, brent_price)
        SpreadService.save_spread_to_db(wti_price,brent_price,spread)

        spread_history = SpreadService.fetch_spread_history()
        """Calculate mean and standard deviation"""
        mean, standard_deviation = calculate_statistics(spread_history)

        """Generate trading signals"""
        signal = generate_signal(spread, mean, standard_deviation)

        return jsonify({
            "spread": spread,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "signal": signal
        })

    except Exception as e:
        logger.error(f"Error in spread analysis API: {e}")
        return jsonify({"error": e}),500




if __name__ == '__main__':
    logger.info("Starting Commodities Quantitative Risk Engine - CQRE")
    app.run(debug=True)
