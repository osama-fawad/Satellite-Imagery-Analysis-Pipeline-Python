from flask import Flask, jsonify, request
import logging
from models.black_scholes import black_scholes

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


if __name__ == '__main__':
    logger.info("Starting Commodities Quantitative Risk Engine - CQRE")
    app.run(debug=True)
