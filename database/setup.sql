CREATE TABLE spread_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    wti_price NUMERIC,
    brent_price NUMERIC,
    spread NUMERIC
);
