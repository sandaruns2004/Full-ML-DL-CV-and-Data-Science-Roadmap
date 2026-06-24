# Analytics Engineering

## What Problem Does This Solve?

Historically, there were two roles in data:
1. **Data Engineers:** Wrote complex Java/Scala pipelines to move raw data from the application database into the Data Warehouse.
2. **Data Analysts / Scientists:** Queried that raw data to build models and dashboards.

The problem? The raw data was incredibly messy. A "User" might exist in the Stripe billing database, the Salesforce CRM, and the PostgreSQL app database, all with slightly different IDs and email formats. 

Data Scientists were spending 80% of their time just cleaning and joining this raw data before they could do any actual science. 

**Analytics Engineering** emerged as a new discipline to sit between Data Engineering and Data Science. It applies strict software engineering principles (version control, testing, CI/CD) to SQL data transformation.

---

## Core Concepts

### 1. ETL vs. ELT
- **ETL (Extract, Transform, Load):** The old way. Data was pulled from the source, transformed on a dedicated server (which was slow), and then loaded into the warehouse.
- **ELT (Extract, Load, Transform):** The modern way. Because modern Data Warehouses (Snowflake, BigQuery) are so incredibly powerful, we just dump the raw data directly into the warehouse (Load), and then use the warehouse's massive compute power to run SQL scripts to clean it (Transform).

### 2. Data Modeling
Raw data is stored in "3rd Normal Form" (highly relational, normalized databases) to make the web app run fast. But this is terrible for analytics, requiring 15 `JOIN` statements just to see a user's revenue.

Analytics Engineers build **Data Models** (like Star Schemas or Snowflake Schemas). They create wide, denormalized tables (e.g., `dim_customers`, `fct_orders`) so that business users and Data Scientists can query them easily without writing complex SQL.

### 3. The Tool: dbt (data build tool)
If you want to be an Analytics Engineer, you must know `dbt`. It is the absolute industry standard.

`dbt` allows you to write simple `SELECT` statements, and it handles all the boilerplate of wrapping them into tables or views. More importantly, it brings Software Engineering to SQL:
- **Version Control:** All SQL is stored in Git.
- **Modularity:** You can reference other SQL files using Jinja templating (e.g., `SELECT * FROM {{ ref('stg_users') }}`).
- **Testing:** You can write automated tests (e.g., "Ensure `user_id` is never null", "Ensure `revenue` is always positive"). If the raw data changes and breaks these rules, dbt stops the pipeline and alerts the team.

---

## Workflow: The Modern Analytics Stack

1. **Fivetran / Airbyte (Extract & Load):** These tools automatically pull data from Stripe, Salesforce, and Postgres and dump it raw into Snowflake (ELT).
2. **dbt (Transform):** The Analytics Engineer writes dbt models to clean the Stripe and Salesforce data, standardize the emails, and join them into a pristine `core_customers` table.
3. **Great Expectations (Test):** Validates the data quality.
4. **Data Science / BI (Consume):** The Data Scientist connects Jupyter to Snowflake and runs `SELECT * FROM core_customers`. The data is perfectly clean.

---

## From Scratch Implementation: A Simple dbt Model

Instead of writing a massive 500-line SQL query, `dbt` breaks it down.

**File 1: `stg_stripe_payments.sql`** (Clean the raw data)
```sql
SELECT
    id as payment_id,
    order_id,
    amount / 100 as amount_dollars, -- Convert cents to dollars
    status
FROM raw.stripe.payments
WHERE status = 'successful'
```

**File 2: `stg_app_orders.sql`** (Clean the app data)
```sql
SELECT
    id as order_id,
    user_id,
    order_date
FROM raw.app_db.orders
```

**File 3: `fct_revenue.sql`** (Join them together cleanly)
```sql
-- Notice the Jinja templating 'ref()' instead of hardcoding table names
SELECT
    o.order_id,
    o.user_id,
    o.order_date,
    p.amount_dollars
FROM {{ ref('stg_app_orders') }} o
LEFT JOIN {{ ref('stg_stripe_payments') }} p
    ON o.order_id = p.order_id
```

If the underlying table structure changes, dbt knows exactly which downstream models are affected via a Directed Acyclic Graph (DAG) and updates them in the correct order.

---

## Common Failure Cases

1. **Spaghetti SQL:** Without dbt, companies accumulate thousands of "Views" in their database. Nobody knows which view depends on which. If you delete a table, 50 dashboards mysteriously break.
2. **Garbage In, Garbage Out:** If the Analytics Engineering layer fails to test for duplicates, the Data Scientist might train a model on duplicated data, ruining the accuracy.
3. **Over-Normalization:** Making the analytics data model too complex. Data scientists just want wide, flat tables to feed into XGBoost.

---

## Industry Applications

- **Spotify:** Processes billions of streaming events. Analytics Engineers model this data into clean "listening sessions" so Data Scientists can build recommendation algorithms without dealing with raw JSON logs.
- **Shopify:** Provides merchants with pristine analytics dashboards. The complex logic joining inventory, shipping, and billing is entirely managed by Analytics Engineering pipelines.

---

## Key Takeaways

1. Analytics Engineering bridges the gap between raw data (Data Engineering) and insights (Data Science).
2. ELT has replaced ETL due to the power of modern cloud data warehouses.
3. `dbt` is the industry standard for transforming data using Software Engineering best practices (Git, testing, modularity).
4. Clean data models free Data Scientists to actually do science.

## Next Topic

Now that the data is modeled perfectly, the models are trained, and the decisions are automated, we must deploy this entirely automated brain to production. How do we ensure it doesn't break when the real world changes?

Navigation:

[← Previous Topic](./13-Data_Product_Thinking.md) | [Back to Index](./README.md) | [Next Topic: Production Data Science →](./15-Production_Data_Science.md)
