import os
import json
import pandas as pd

path = r"C:/Users/icon/Downloads/Labmentix Project 4/Phone Pe/pulse-master/data/aggregated/transaction/country/india/state/"

data_list = []

for state in os.listdir(path):
    state_path = os.path.join(path, state)
    
    for year in os.listdir(state_path):
        year_path = os.path.join(state_path, year)
        
        for file in os.listdir(year_path):
            if file.endswith(".json"):
                file_path = os.path.join(year_path, file)
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    if data.get("data") and data["data"].get("transactionData"):
                        for item in data["data"]["transactionData"]:
                            data_list.append({
                                "State": state,
                                "Year": int(year),
                                "Quarter": int(file.replace(".json", "")),
                                "Type": item["name"],
                                "Count": item["paymentInstruments"][0]["count"],
                                "Amount": item["paymentInstruments"][0]["amount"]
                            })

df = pd.DataFrame(data_list)

print(df.head())

df.to_csv("phonepe_transaction.csv", index=False)

print("Done ✅")