DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_metrics;

CREATE TABLE customers (
  customerid TEXT PRIMARY KEY,
  gender TEXT,
  seniorcitizen INTEGER,
  partner TEXT,
  dependents TEXT,
  tenure INTEGER,
  phoneservice TEXT,
  multiplelines TEXT,
  internetservice TEXT,
  onlinesecurity TEXT,
  onlinebackup TEXT,
  deviceprotection TEXT,
  techsupport TEXT,
  streamingtv TEXT,
  streamingmovies TEXT,
  contract TEXT,
  paperlessbilling TEXT,
  paymentmethod TEXT,
  monthlycharges REAL,
  totalcharges REAL,
  churn TEXT
);

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  customerid TEXT,
  model_version TEXT,
  churn_probability REAL,
  churn_prediction INTEGER,
  created_at TEXT,
  FOREIGN KEY (customerid) REFERENCES customers(customerid)
);

CREATE TABLE model_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_version TEXT,
  metric_name TEXT,
  metric_value REAL,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS model_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    churn_probability REAL NOT NULL,
    churn_prediction INTEGER NOT NULL,
    created_at NOT NULL 
);
