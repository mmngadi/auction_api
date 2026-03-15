-- init.sql
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS auction_lots;

CREATE TABLE auction_lots (
    id SERIAL PRIMARY KEY,
    status_of_lot TEXT,
    winning_bid_amount INTEGER,
    date_of_auction DATE,
    year INTEGER,
    make TEXT,
    category TEXT,
    trade_amount TEXT,
    mileage TEXT,
    service_book TEXT,
    no_of_keys INTEGER,
    colour TEXT,
    condition TEXT,
    condition_embedding vector(768)
);

COPY auction_lots(
    status_of_lot,
    winning_bid_amount,
    date_of_auction,
    year,
    make,
    category,
    trade_amount,
    mileage,
    service_book,
    no_of_keys,
    colour,
    condition
)
FROM '/docker-entrypoint-initdb.d/dataset.csv'
DELIMITER ','
CSV HEADER;
