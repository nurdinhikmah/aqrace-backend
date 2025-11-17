from urllib.parse import urlparse, unquote
import pandas as pd
import math
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
domain_freq_path = os.path.join(BASE_DIR, "domain_freq.pkl")
domain_freq = joblib.load(domain_freq_path)

# Helper: check if hostname is IPv4
def is_ipv4_host(hostname):
    if not hostname:
        return False
    parts = hostname.split('.')
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except ValueError:
        return False

# Helper: check for homoglyphs (e.g., Cyrillic or Greek letters disguised as Latin)
def has_homoglyphs(text):
    return any(ord(ch) > 127 for ch in text)

# Suspicious keywords commonly used in phishing
SUSPICIOUS_KEYWORDS = [
    "login", "secure", "update", "verify", "account",
    "banking", "signin", "confirm", "password", "pay"
]

# Brand impersonation keywords
BRAND_KEYWORDS = [
    "maybank", "cimb", "bankislam", "rhb", "bni", "bri", "bca",
    "tng", "touchngo", "shopee", "lazada",
    "facebook", "instagram", "tiktok", "google", "youtube", "gmail",
    "paypal", "amazon"
]

# Shannon Entropy for randomness detection
def shannon_entropy(s):
    freq = {ch: s.count(ch) for ch in set(s)}
    total = len(s)
    return -sum((c/total) * math.log2(c/total) for c in freq.values()) if total > 0 else 0


def extract_url_features(urls):
    if isinstance(urls, str):
        urls = [urls]

    data = []

    for url in urls:
        try:
            url_unquoted = unquote(url)
            parsed = urlparse(url_unquoted)

            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower().split('@')[-1]
            hostname = netloc.split(':')[0]
            path = parsed.path or ""

            # === Basic Features ===
            url_length = len(url_unquoted)
            domain_length = len(hostname)
            path_length = len(path)
            num_subdomain = max(0, hostname.count('.') - 1)
            is_ip_address = 1 if is_ipv4_host(hostname) else 0

            # === Character Features ===
            num_special_chars = sum(1 for ch in url_unquoted if not ch.isalnum())
            num_digits = sum(ch.isdigit() for ch in url_unquoted)
            has_https = 1 if scheme == 'https' else 0

            # === Suspicious indicators ===
            has_suspicious_keyword = 1 if any(k in url_unquoted.lower() for k in SUSPICIOUS_KEYWORDS) else 0
            has_homoglyph = 1 if has_homoglyphs(url_unquoted) else 0

            # === Extra Features (Fast Accuracy Boost) ===
            entropy = shannon_entropy(url_unquoted)
            contains_brand = 1 if any(b in url_unquoted.lower() for b in BRAND_KEYWORDS) else 0

            # === Domain Frequency Feature ===
            domain_frequency = domain_freq.get(hostname, 1)

            data.append({
                "URLLength": url_length,
                "DomainLength": domain_length,
                "PathLength": path_length,
                "NumSubdomain": num_subdomain,
                "IsIPAddress": is_ip_address,
                "NumSpecialChar": num_special_chars,
                "NumDigits": num_digits,
                "HasHTTPS": has_https,
                "HasSuspiciousKeyword": has_suspicious_keyword,
                "HasHomoglyphs": has_homoglyph,
                "URL_Entropy": entropy,
                "ContainsBrandKeyword": contains_brand,
                "DomainFrequency": domain_frequency
            })

        except Exception:
            continue

    return pd.DataFrame(data)
