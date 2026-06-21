# 🎣 Data Collection

> **Prerequisites**: [Python for Data Science](./02-Python-for-Data-Science.md) | **Difficulty**: ⭐☆☆☆☆ Beginner to 🟡 Intermediate

---

## 📋 Table of Contents

1. [Where Does Data Come From?](#1-where-does-data-come-from)
2. [Reading Flat Files (CSV, Excel)](#2-reading-flat-files-csv-excel)
3. [Working with APIs (JSON)](#3-working-with-apis-json)
4. [Web Scraping (HTML)](#4-web-scraping-html)
5. [Connecting to Databases (SQL)](#5-connecting-to-databases-sql)
6. [What's Next](#6-whats-next)

---

## 1. Where Does Data Come From?

### 🟢 Beginner

In Kaggle competitions or university courses, data is usually handed to you as a clean `.csv` file. In the real world, **Data Scientists must hunt for data.**

The four main sources of data:
1. **Flat Files**: Spreadsheets, CSVs, text logs stored locally or in cloud storage (S3).
2. **APIs**: Communicating with other software over the internet (e.g., getting weather data from a weather service).
3. **Web Scraping**: Extracting data directly from websites that don't have an API.
4. **Databases**: Querying your company's internal SQL/NoSQL databases.

---

## 2. Reading Flat Files (CSV, Excel)

### 🟢 Beginner

Pandas handles flat files exceptionally well.

**Reading CSV (Comma-Separated Values)**
```python
import pandas as pd

# The simplest way
df = pd.read_csv('sales_data.csv')

# Handling common messy CSV issues
df = pd.read_csv(
    'sales_data.csv',
    sep=';',                 # Sometimes European datasets use semicolons
    parse_dates=['Date'],    # Automatically convert date strings to datetime objects
    index_col='ID',          # Use the 'ID' column as the row index
    na_values=['NA', 'missing', '?'] # Treat these strings as NaN (Not a Number)
)
```

### 🟡 Intermediate

**Reading Excel Files**

Excel files `.xlsx` can have multiple sheets. You need `openpyxl` installed (`pip install openpyxl`).

```python
import pandas as pd

# Read a specific sheet
df_q1 = pd.read_excel('financials.xlsx', sheet_name='Q1_Revenue')

# Read multiple sheets into a dictionary of DataFrames
all_sheets = pd.read_excel('financials.xlsx', sheet_name=['Q1', 'Q2'])
df_q2 = all_sheets['Q2']
```

**Memory Optimization for Huge CSVs**
If a CSV is 5GB but you only have 8GB of RAM, you can't load it all at once!

```python
# 1. Read only specific columns
columns_to_keep = ['user_id', 'purchase_amount']
df = pd.read_csv('huge_file.csv', usecols=columns_to_keep)

# 2. Read in chunks (Generator approach)
chunk_size = 100_000
total_revenue = 0

# This loop reads 100k rows, processes them, dumps them from RAM, and gets the next 100k
for chunk_df in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    total_revenue += chunk_df['purchase_amount'].sum()

print(f"Total Revenue: ${total_revenue}")
```

---

## 3. Working with APIs (JSON)

### 🟡 Intermediate

An **API** (Application Programming Interface) allows your Python code to talk to a web server. Most web APIs return data in **JSON** (JavaScript Object Notation) format, which looks exactly like Python dictionaries.

You will need the `requests` library (`pip install requests`).

```python
import requests
import pandas as pd

# 1. Make a GET request to a public API
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
response = requests.get(url)

# 2. Check if the request was successful (Status Code 200 means OK)
if response.status_code == 200:
    # 3. Parse the JSON data into a Python dictionary
    data = response.json()
    print("Raw JSON:", data) 
    # Output: {'bitcoin': {'usd': 65000}, 'ethereum': {'usd': 3500}}
    
    # 4. Convert to Pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    print("\nDataFrame:\n", df)
else:
    print(f"Error! Status Code: {response.status_code}")
```

### 🔴 Advanced

**Handling API Rate Limits and Authentication**

Enterprise APIs require API keys and limit how fast you can pull data.

```python
import requests
import time

headers = {
    "Authorization": "Bearer YOUR_SECRET_API_KEY",
    "Accept": "application/json"
}

# Implementing a backoff strategy for rate limits
def fetch_data_safely(url):
    for attempt in range(3):  # Try 3 times
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429: # 429 = Too Many Requests
            print("Rate limit hit. Waiting 5 seconds...")
            time.sleep(5)
        else:
            print(f"Failed: {response.status_code}")
            break
            
    return None
```

---

## 4. Web Scraping (HTML)

### 🟡 Intermediate

When a website doesn't have an API, you must download the HTML and extract the data manually. This requires `beautifulsoup4` and `requests`.

> ⚠️ **Ethics Note**: Always check a website's `robots.txt` (e.g., `www.website.com/robots.txt`) before scraping. Do not overload servers by scraping too fast.

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://quotes.toscrape.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all 'div' elements with the class 'quote'
quotes_data = []
quote_elements = soup.find_all('div', class_='quote')

for element in quote_elements:
    # Extract the text and author
    text = element.find('span', class_='text').text
    author = element.find('small', class_='author').text
    
    quotes_data.append({
        'Quote': text,
        'Author': author
    })

df = pd.DataFrame(quotes_data)
print(df.head())
```

### 🔴 Advanced

Many modern websites use JavaScript to render data. `requests` + `BeautifulSoup` won't work because they don't execute JavaScript. 
For advanced scraping, you must use browser automation tools like **Selenium** or **Playwright**, which actually open a hidden Chrome browser, wait for the JS to load, and then extract the data.

---

## 5. Connecting to Databases (SQL)

### 🟡 Intermediate

In a corporate environment, your data lives in SQL databases (PostgreSQL, MySQL, Snowflake).
Python connects to these using `SQLAlchemy`.

```python
import pandas as pd
from sqlalchemy import create_engine

# 1. Create a connection engine
# Format: dialect://username:password@host:port/database
engine = create_engine('postgresql://user:password@localhost:5432/my_company_db')

# 2. Write your SQL query
query = """
    SELECT user_id, signup_date, total_spent
    FROM customers
    WHERE total_spent > 1000
    ORDER BY total_spent DESC
"""

# 3. Read directly into Pandas!
try:
    df = pd.read_sql(query, con=engine)
    print(df.head())
except Exception as e:
    print("Database connection failed:", e)
```

---

## 6. What's Next

Once you have acquired your data, you will immediately notice something: **it is messy**. 

| Next Topic | Why |
|------------|-----|
| [Data Cleaning](./04-Data-Cleaning.md) | Learn how to fix missing values, correct data types, and handle duplicates so your models don't crash. |

---

[← Previous: Python for Data Science](./02-Python-for-Data-Science.md) | [Back to Main Index](../README.md) | [Next: Data Cleaning →](./04-Data-Cleaning.md)
