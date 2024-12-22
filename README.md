# Commodities Quantitative Risk Engine (CQRE)

## Overview
The **Commodities Quantitative Risk Engine (CQRE)** is a robust backend system designed to analyze real-time and historical data of WTI and Brent crude oil prices. The system employs advanced financial models such as **Black-Scholes Option Pricing** and **Mean Reversion Spread Analysis** to generate actionable trading signals and evaluate strategy performance.

This project demonstrates:
- Integration of **real-time data processing** using Kafka.
- Advanced financial modeling with Python.
- API-driven architecture for analytical insights.
- Scalable backend setup for commodities trading.

---

## Features
1. **Real-Time Data Analysis**:
   - Fetches live WTI and Brent prices through a Kafka producer.
   - Processes the prices via a Kafka consumer to calculate spreads and generate trading signals.

2. **Financial Models**:
   - **Black-Scholes Option Pricing**: Calculates the fair value of options based on WTI/Brent prices.
   - **Mean Reversion Analysis**: Detects price deviations and generates buy/sell signals.

3. **APIs**:
   - **Historical Spread Data**: Fetch stored spread data from PostgreSQL.
   - **Black-Scholes Pricing**: API to calculate the price of options.
   - **Mean Reversion**: API to use the mean reversion strategy.

4. **Data Storage**:
   - Stores real-time price and spread data in a PostgreSQL database for historical analysis.

---

## Technology Stack
- **Programming Language**: Python
- **Message Broker**: Apache Kafka
- **Database**: PostgreSQL
- **Backend Framework**: Flask
- **Data Models**:
  - **Black-Scholes Option Pricing** for financial analysis.
  - **Mean Reversion Spread Analysis** for trading signals.

---

## APIs
### 1. **Historical Spread Data**
   - **Endpoint**: `GET /api/historical-spreads`
   - **Description**: Fetch historical WTI and Brent price spreads.
   - **Response**:
     ```json
     [
       {"timestamp": "2024-12-08T10:00:00", "wti_price": 75.32, "brent_price": 78.54, "spread": 3.22},
       {"timestamp": "2024-12-08T10:05:00", "wti_price": 76.10, "brent_price": 79.80, "spread": 3.70}
     ]
     ```

### 2. **Black-Scholes Pricing**
   - **Endpoint**: `POST /api/option-pricing`
   - **Description**: Calculate the fair value of an option using Black-Scholes.
   - **Request**:
     ```json
     {
       "option_type": "call",
       "S": 75.0,
       "K": 80.0,
       "T": 0.5,
       "r": 0.02,
       "sigma": 0.3
     }
     ```
   - **Response**:
     ```json
     {
       "price": 5.34
     }
     ```

---

## How It Works
1. **Kafka Producer**:
   - Produces random or real-time WTI and Brent prices every 1 second.
2. **Kafka Consumer**:
   - Consumes prices, calculates spreads, stores data in PostgreSQL, and generates trading signals.
3. **API Layer**:
   - Exposes historical data, Mean Reversion and  Black-Scholes calculations via Flask endpoints.

---

## Requirements
- **Python**: Version 3.11 or higher.
- **Kafka**: Apache Kafka should be installed and running on `localhost:9092`.
- **PostgreSQL**: Database should be running on `localhost:5432`.
- **Python Packages**:
  - Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## Setup and Run Instructions
### 1. **Install Prerequisites**
   - Install Python 3.11 or higher.
   - Install PostgreSQL and create a database:
     ```sql
     CREATE DATABASE cqre;
     CREATE USER cqre WITH PASSWORD 'cqre';
     GRANT ALL PRIVILEGES ON DATABASE cqre TO cqre;
     ```

   - Start Kafka:
     ```bash
     kafka-server-start.sh config/server.properties
     ```

### 2. **Set Environment Variables**
   Create a `.env` file or export the following:
   ```bash
   DATABASE_URL=postgresql://cqre:cqre@localhost:5432/cqre
   KAFKA_BOOTSTRAP_SERVERS=localhost:9092
