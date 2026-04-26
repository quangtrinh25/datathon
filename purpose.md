```markdown
# DATATHON 2026 – ROUND 1: SALES FORECASTING  
## Problem Statement & Data Description

This document provides a complete and detailed description of the **Sales Forecasting** task in Round 1 of DATATHON 2026. It covers the business context, problem definition, evaluation metrics, submission format, constraints, and the full data dictionary with all tables, columns, relationships, and important notes.

This markdown file contains **only** the problem description and data specifications. No modeling methods, solution approaches, or code suggestions are included.

---

## 1. Business Context

You are a data scientist working for a Vietnamese e-commerce fashion company. Accurate daily revenue and COGS forecasting is essential for optimizing inventory allocation, planning promotions, managing logistics, and supporting strategic business decisions across the country.

---

## 2. Problem Definition

**Objective:**  
Forecast the daily **Revenue** and **COGS** for the period covered by the hidden test set (`sales_test.csv`).

The task is a **daily time series forecasting** problem where you must predict two target variables for each future date:

- `Revenue`: Net daily revenue after discounts and returns.
- `COGS`: Total cost of goods sold for that day.

### Data Splits

| Split       | File                | Time Period                  | Usage                          |
|-------------|---------------------|------------------------------|--------------------------------|
| Train       | `sales.csv`         | 2012-07-04 → 2022-12-31      | Training & cross-validation    |
| Test (hidden) | `sales_test.csv`  | 2023-01-01 → 2024-07-01      | Evaluation only                |

**Note:** The actual test file (`sales_test.csv`) is **not provided** to participants. It is used internally on the evaluation platform. The structure and dates match `sample_submission.csv`.

---

## 3. Evaluation Metrics

Submissions are evaluated using the following three metrics on both **Revenue** and **COGS** predictions (typically averaged or reported separately depending on platform rules):

### 3.1. Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n} \sum_{i=1}^{n} |F_i - A_i|
\]

### 3.2. Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (F_i - A_i)^2}
\]

### 3.3. Coefficient of Determination (R²)

\[
R^2 = 1 - \frac{\sum_{i=1}^{n} (A_i - F_i)^2}{\sum_{i=1}^{n} (A_i - \bar{A})^2}
\]

Where:
- \( F_i \): Predicted (forecasted) value
- \( A_i \): Actual value
- \( \bar{A} \): Mean of actual values
- \( n \): Number of observations

**Goal:** Minimize MAE and RMSE while maximizing R².

---

## 4. Submission Format

Participants must submit a file named **`submission.csv`** with exactly the following structure and columns:

```csv
Date,Revenue,COGS
2023-01-01,0.0,0.0
2023-01-02,0.0,0.0
...
2024-07-01,0.0,0.0
```

- The file must contain **exactly** the same number of rows and in the **exact same order** as `sample_submission.csv`.
- Do **not** change the order of rows, add or remove any dates.
- Only the values in the `Revenue` and `COGS` columns should be replaced with your predictions.

---

## 5. Constraints

- **No external data**: All features must be derived exclusively from the provided datasets.
- **No look-ahead bias**: You cannot use any information from future dates when creating features for a specific forecast date.
- **Reproducibility**: All source code must be submitted. Random seeds must be set where applicable so results can be reproduced.
- **Explainability**: The technical report must include a clear section (in business language) explaining the main drivers of the revenue and COGS forecasts (e.g., feature importances, SHAP values, or similar interpretable outputs).
- **Disqualification risks**: The forecasting section will receive zero points if any of the following occurs:
  - Using test set `Revenue` or `COGS` values as features.
  - Using data outside the provided dataset.
  - Failure to submit reproducible code.

---

## 6. Detailed Data Dictionary

### 6.1. Core Forecasting Data

#### `sales.csv` – Training Revenue Data (Daily)

| Column     | Type   | Description                                              |
|------------|--------|----------------------------------------------------------|
| `Date`     | date   | Transaction date (primary time axis).                    |
| `Revenue`  | float  | Net daily revenue (after discounts and returns).         |
| `COGS`     | float  | Total cost of goods sold for the day.                    |

- **Period**: 2012-07-04 to 2022-12-31
- **Frequency**: Daily (some dates may be missing — check for gaps)

#### `sales_test.csv` – Hidden Test Set

| Column     | Type   | Description                                              |
|------------|--------|----------------------------------------------------------|
| `Date`     | date   | Date to forecast (2023-01-01 to 2024-07-01).             |
| `Revenue`  | float  | Target variable (hidden).                                |
| `COGS`     | float  | Target variable (hidden).                                |

#### `sample_submission.csv` – Submission Template

Contains the complete list of dates in the test period in the required order. Use this file as the base for your `submission.csv`.

---

### 6.2. Master Data (Reference Tables)

#### `products.csv` – Product Catalog

| Column         | Type    | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `product_id`   | int     | Primary key                                              |
| `product_name` | str     | Product name                                             |
| `category`     | str     | Product category (e.g., Streetwear, Sport, ...)          |
| `segment`      | str     | Market segment (Premium, Performance, Activewear, Standard) |
| `size`         | str     | Size (S, M, L, XL, ...)                                  |
| `color`        | str     | Color label                                              |
| `price`        | float   | Retail list price                                        |
| `cogs`         | float   | Cost of goods sold per unit                              |

**Constraint**: `cogs < price` for all products.

#### `customers.csv` – Customer Information

| Column                | Type    | Description                                              |
|-----------------------|---------|----------------------------------------------------------|
| `customer_id`         | int     | Primary key                                              |
| `zip`                 | int     | Postal code (FK → geography.zip)                         |
| `city`                | str     | City                                                     |
| `signup_date`         | date    | Account registration date                                |
| `gender`              | str     | Gender (nullable)                                        |
| `age_group`           | str     | Age group (nullable)                                     |
| `acquisition_channel` | str     | Marketing channel at signup (nullable)                   |

#### `promotions.csv` – Promotion Campaigns

| Column                | Type    | Description                                              |
|-----------------------|---------|----------------------------------------------------------|
| `promo_id`            | str     | Primary key                                              |
| `promo_name`          | str     | Campaign name + year                                     |
| `promo_type`          | str     | `percentage` or `fixed`                                  |
| `discount_value`      | float   | Discount value (% or fixed amount)                       |
| `start_date`          | date    | Campaign start date                                      |
| `end_date`            | date    | Campaign end date                                        |
| `applicable_category` | str     | Category applied to (NULL = all categories)              |
| `promo_channel`       | str     | Channel where promotion applies (nullable)               |
| `stackable_flag`      | int     | 1 = can be combined, 0 = cannot                          |
| `min_order_value`     | float   | Minimum order value to apply (nullable)                  |

**Discount Formulas:**
- Percentage: `discount_amount = quantity × unit_price × (discount_value / 100)`
- Fixed: `discount_amount = quantity × discount_value`

#### `geography.csv` – Geographic Mapping

| Column     | Type | Description                     |
|------------|------|---------------------------------|
| `zip`      | int  | Primary key (postal code)       |
| `city`     | str  | City name                       |
| `region`   | str  | Region (West, Central, East)    |
| `district` | str  | District name                   |

---

### 6.3. Transaction Data (Detailed Transactions)

#### `orders.csv` – Order Header

| Column           | Type   | Description                                              |
|------------------|--------|----------------------------------------------------------|
| `order_id`       | int    | Primary key                                              |
| `order_date`     | date   | Order date (FK → sales.Date)                             |
| `customer_id`    | int    | FK → customers.customer_id                               |
| `zip`            | int    | Shipping postal code (FK → geography.zip)                |
| `order_status`   | str    | Status: delivered, shipped, cancelled, returned, ...     |
| `payment_method` | str    | Payment method                                           |
| `device_type`    | str    | Device: mobile, desktop, tablet                          |
| `order_source`   | str    | Marketing channel that led to the order                  |

#### `order_items.csv` – Order Line Items

| Column            | Type    | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| `order_id`        | int     | FK → orders.order_id                                     |
| `product_id`      | int     | FK → products.product_id                                 |
| `quantity`        | int     | Quantity ordered                                         |
| `unit_price`      | float   | Unit price after promotions                              |
| `discount_amount` | float   | Total discount for this line item                        |
| `promo_id`        | str     | FK → promotions.promo_id (nullable)                     |
| `promo_id_2`      | str     | Second promotion if stackable (nullable)                 |

#### `payments.csv` – Payment Information (1:1 with orders)

| Column            | Type    | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| `order_id`        | int     | FK → orders.order_id                                     |
| `payment_method`  | str     | Payment method                                           |
| `payment_value`   | float   | Total payment amount for the order                       |
| `installments`    | int     | Number of installment payments                           |

#### `shipments.csv` – Shipping Information

| Column          | Type    | Description                                              |
|-----------------|---------|----------------------------------------------------------|
| `order_id`      | int     | FK → orders.order_id                                     |
| `ship_date`     | date    | Date order was shipped                                   |
| `delivery_date` | date    | Date order was delivered                                 |
| `shipping_fee`  | float   | Shipping fee (0 if waived)                               |

**Note:** Only exists for orders with status `shipped`, `delivered`, or `returned`.

#### `returns.csv` – Return Records

| Column            | Type    | Description                                              |
|-------------------|---------|----------------------------------------------------------|
| `return_id`       | str     | Primary key                                              |
| `order_id`        | int     | FK → orders.order_id                                     |
| `product_id`      | int     | FK → products.product_id                                 |
| `return_date`     | date    | Date customer returned the item                          |
| `return_reason`   | str     | Reason: defective, wrong_size, changed_mind, not_as_described |
| `return_quantity` | int     | Quantity returned                                        |
| `refund_amount`   | float   | Amount refunded                                          |

#### `reviews.csv` – Product Reviews

| Column         | Type    | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `review_id`    | str     | Primary key                                              |
| `order_id`     | int     | FK → orders.order_id                                     |
| `product_id`   | int     | FK → products.product_id                                 |
| `customer_id`  | int     | FK → customers.customer_id                               |
| `review_date`  | date    | Date review was submitted                                |
| `rating`       | int     | Rating from 1 to 5                                       |
| `review_title` | str     | Review title                                             |

---

### 6.4. Operational Data

#### `inventory.csv` – Monthly Inventory Snapshot (End of Month)

| Column              | Type    | Description                                              |
|---------------------|---------|----------------------------------------------------------|
| `snapshot_date`     | date    | Last day of the month                                    |
| `product_id`        | int     | FK → products.product_id                                 |
| `stock_on_hand`     | int     | Stock quantity at month end                              |
| `units_received`    | int     | Quantity received during the month                       |
| `units_sold`        | int     | Quantity sold during the month                           |
| `stockout_days`     | int     | Number of days out of stock                              |
| `days_of_supply`    | float   | Days of supply remaining                                 |
| `fill_rate`         | float   | Fraction of orders fully filled from stock               |
| `stockout_flag`     | int     | 1 if stockout occurred                                   |
| `overstock_flag`    | int     | 1 if overstock occurred                                  |
| `reorder_flag`      | int     | 1 if reorder is recommended                              |
| `sell_through_rate` | float   | Sell-through rate                                        |
| `product_name`      | str     | Product name (redundant)                                 |
| `category`          | str     | Category                                                 |
| `segment`           | str     | Segment                                                  |
| `year`              | int     | Year                                                     |
| `month`             | int     | Month                                                    |

**Important:** This is a monthly end-of-month snapshot. When creating daily features, be careful to avoid look-ahead bias.

#### `web_traffic.csv` – Daily Website Traffic

| Column                    | Type    | Description                                              |
|---------------------------|---------|----------------------------------------------------------|
| `date`                    | date    | Date of record                                           |
| `sessions`                | int     | Total sessions                                           |
| `unique_visitors`         | int     | Unique visitors                                          |
| `page_views`              | int     | Total page views                                         |
| `bounce_rate`             | float   | Bounce rate                                              |
| `avg_session_duration_sec`| float   | Average session duration (seconds)                       |
| `conversion_rate`         | float   | Conversion rate (sessions → orders)                      |
| `traffic_source`          | str     | Source: organic_search, paid_search, email_campaign, social_media |

**Frequency:** Daily, aligned with `sales.csv`.

---

## 7. Logical Relationships Between Tables

```
sales.Date ──────────────────────────────┬── web_traffic.date (1:1)
                                         │
                                         └── orders.order_date (1:N)
                                              │
                                              ├── order_items (N:1) ── products (N:1)
                                              │        │
                                              │        └── promotions (N:1)
                                              │
                                              ├── payments (1:1)
                                              ├── shipments (1:1)
                                              ├── returns (1:N) ── products
                                              └── reviews (1:N) ── products, customers

customers ── geography (N:1) via zip
```

**Notes:**
- Not all dates in the historical period necessarily have transactions (Revenue may be zero or the row may be missing — you must check).
- `promotions`, `web_traffic`, and `inventory` provide valuable exogenous information.

---

## 8. Special Considerations for Forecasting

- **Time Series Characteristics**: The training data spans more than 10 years of daily observations. Seasonality (monthly, quarterly, yearly) and long-term trends are likely present.
- **Look-ahead Bias**: Strictly prohibited. For example, you cannot use an inventory snapshot from January 2023 to forecast revenue in December 2022.
- **Net Revenue**: The `Revenue` column in `sales.csv` is already net of discounts and returns. The goal is to forecast this value directly.
- **COGS Forecasting**: You must predict both `Revenue` and `COGS`. While they are correlated, the relationship is not fixed due to changing product mix.
- **Missing Dates**: Check whether dates with zero activity are missing or present with zero values.
- **Explainability**: The solution must allow clear identification of the most influential factors (e.g., day of week, traffic, promotions, etc.) in business terms.
```
