-- Active: 1770487880142@@127.0.0.1@5432@fintech
CREATE TABLE transactions (
    step INT,
    type TEXT,
    amount FLOAT,
    oldbalanceOrg FLOAT,
    newbalanceOrg FLOAT,
    fraud INT
);

SELECT * FROM transactions;