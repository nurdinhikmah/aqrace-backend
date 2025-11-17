
import pandas as pd
from urllib.parse import urlparse, urlunparse, unquote
from sklearn.utils import resample

# ----------------------
# 1) File paths
# ----------------------
a = '/Users/imma/Desktop/Backend/balanced_urls.csv'        # url,label,result
b = '/Users/imma/Desktop/Backend/malicious_phish copy.csv' # url,type

# ----------------------
# 2) Load (standardise column names)
# ----------------------
df_a = pd.read_csv(a)
df_b = pd.read_csv(b)

# Normalize column names to lowercase
df_a.columns = [c.lower() for c in df_a.columns]
df_b.columns = [c.lower() for c in df_b.columns]

# Ensure url column exists
if 'url' not in df_a.columns or 'url' not in df_b.columns:
    raise SystemExit("Missing 'url' column in one of the files")

# Create a unified label column:
# df_a likely has 'label' with 'benign' values -> keep as-is
# df_b has 'type' with values like 'phishing', 'benign', 'malicious', 'defacement'
# Map: anything NOT 'benign' -> 'malicious'
df_a['label'] = df_a.get('label', df_a.get('type', '')).astype(str).str.lower().str.strip()
df_b['label'] = df_b.get('type', df_b.get('label', '')).astype(str).str.lower().str.strip()
df_b['label'] = df_b['label'].apply(lambda x: x if x == 'benign' else 'malicious')

# If df_a has numeric 'result' column, prefer textual label already present.
# For safety, also map any non-benign in df_a to malicious
df_a['label'] = df_a['label'].apply(lambda x: x if x == 'benign' else 'malicious')

# Keep only needed columns
df_a = df_a[['url', 'label']].copy()
df_b = df_b[['url', 'label']].copy()

# ----------------------
# 3) Concat datasets
# ----------------------
df = pd.concat([df_a, df_b], ignore_index=True)
df = df.dropna(subset=['url'])
df['url'] = df['url'].astype(str).str.strip()

# ----------------------
# 4) Normalise URL (simple, consistent)
#    - ensure scheme (default http if missing)
#    - unquote, lowercase host, remove leading www., strip fragment, remove trailing slash
# ----------------------
def normalise_url(raw):
    try:
        s = unquote(raw.strip())
        # ensure scheme
        if '://' not in s:
            s = 'http://' + s
        p = urlparse(s)
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        path = p.path.rstrip('/')
        # optionally keep query but sort query params (simple approach: keep as-is)
        # drop fragment
        norm = urlunparse((scheme, netloc, path or '', '', p.query or '', ''))
        return norm
    except Exception:
        return raw.strip().lower()

df['normalized_url'] = df['url'].apply(normalise_url)

# ----------------------
# 5) Extract domain & tld for later sampling controls
# ----------------------
def get_domain(norm_url):
    try:
        return urlparse(norm_url).netloc.lower()
    except:
        return ''

df['domain'] = df['normalized_url'].apply(get_domain)
df['tld'] = df['domain'].apply(lambda d: d.split('.')[-1] if d and '.' in d else d)

# ----------------------
# 6) Deduplicate by normalized URL
# ----------------------
df = df.drop_duplicates(subset=['normalized_url']).reset_index(drop=True)

# ----------------------
# 7) Optional: limit per domain to avoid domination (e.g. max 200 per domain)
# ----------------------
MAX_PER_DOMAIN = 200
df = df.groupby('domain', group_keys=False).apply(lambda g: g.sample(n=min(len(g), MAX_PER_DOMAIN), random_state=42))
df = df.reset_index(drop=True)

# ----------------------
# 8) Optional: limit per TLD to keep geographic diversity (e.g. max 50k per TLD)
# ----------------------
MAX_PER_TLD = 50000
df = df.groupby('tld', group_keys=False).apply(lambda g: g.sample(n=min(len(g), MAX_PER_TLD), random_state=42))
df = df.reset_index(drop=True)

# ----------------------
# 9) Map labels to final form: 'SAFE' and 'MALICIOUS'
#    (we already set non-benign -> malicious earlier; double-check)
# ----------------------
df['label'] = df['label'].apply(lambda x: 'malicious' if x != 'benign' else 'benign')
# Now final mapping
df['final_label'] = df['label'].map({'benign': 'SAFE', 'malicious': 'MALICIOUS'})

# ----------------------
# 10) Balance classes to target (250k each or less if not available)
# ----------------------
ben = df[df['final_label'] == 'SAFE']
mal = df[df['final_label'] == 'MALICIOUS']

target_n = min(len(ben), len(mal), 250000)  # choose up to 250k per class

ben_s = resample(ben, replace=False, n_samples=target_n, random_state=42) if len(ben) >= target_n else ben
mal_s = resample(mal, replace=False, n_samples=target_n, random_state=42) if len(mal) >= target_n else mal

final = pd.concat([ben_s, mal_s]).sample(frac=1, random_state=42).reset_index(drop=True)

# ----------------------
# 11) Save final CSV (only keep url + final label)
# ----------------------
out_path = '/Users/imma/Desktop/Backend/final_SAFE_MALICIOUS_500k.csv'
final[['normalized_url', 'final_label', 'domain']].rename(columns={'normalized_url':'url','final_label':'label'}).to_csv(out_path, index=False)

print("Saved:", out_path)
print("Final shape:", final.shape)
print(final['label'].value_counts())
print("\nTop domains (final):")
print(final['domain'].value_counts().head(20))
