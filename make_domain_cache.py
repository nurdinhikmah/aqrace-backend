import pandas as pd
import joblib

csv_path = "/Users/imma/Desktop/Backend/final_SAFE_MALICIOUS_500k.csv"
df = pd.read_csv(csv_path)

df['domain'] = df['url'].apply(lambda x: x.split('/')[2] if '://' in x else x)
domain_freq = df['domain'].value_counts().to_dict()

joblib.dump(domain_freq, "/Users/imma/Desktop/Backend/domain_freq.pkl")

print("✅ Domain frequency cache created successfully.")
print("Total unique domains:", len(domain_freq))
