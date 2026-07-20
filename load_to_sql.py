import pandas as pd
import mysql.connector

# CSV file load करो
df = pd.read_csv("phonepe_transaction.csv")

# MySQL connect करो
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345",   # 👉 अपना password डालो
    database="phonepe"
)

cursor = conn.cursor()

# Data insert करो
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO aggregated_transaction 
        (state, year, quarter, type, count, amount)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()

print("✅ Data Inserted Successfully")

