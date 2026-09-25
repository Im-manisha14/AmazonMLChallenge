"""
preprocessing.py - Step 2: Text Normalization
=============================================
Robust normalization for business names and addresses.

Key idea: The SAME real business may appear as:
  - "ABC Corp" vs "A.B.C. Corporation"
  - "MG Road" vs "Mahatma Gandhi Road"
  - "Pvt Ltd" vs "Private Limited"

We normalize both sides before comparing so these look more similar.
"""

import re
import unicodedata
import pandas as pd
import numpy as np

# ─── Abbreviation Dictionaries ────────────────────────────────────────────────

# Business name legal suffix abbreviations (expand for normalization)
# These are common shortened forms found in business names
NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "lp": "limited partnership",
    # French legal forms (test data has France)
    "sarl": "societe a responsabilite limitee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "sas": "societe par actions simplifiee",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "sa": "societe anonyme",
    "sci": "societe civile immobiliere",
    # Common words
    "intl": "international",
    "tech": "technology",
    "technologies": "technology",
    "mfg": "manufacturing",
    "grp": "group",
    "mgmt": "management",
    "serv": "services",
    "svc": "services",
    "svcs": "services",
    "assoc": "associates",
    "dept": "department",
    "univ": "university",
    "ctr": "center",
    "hosp": "hospital",
    "engr": "engineering",
    "ent": "enterprises",
    "ind": "industries",
    "mkt": "market",
    "mkts": "markets",
    "sys": "systems",
    "dev": "development",
    "natl": "national",
    "natl": "national",
    "bros": "brothers",
}

# Address abbreviations (street types, directions, etc.)
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "cir": "circle",
    "fl": "floor",
    "flr": "floor",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "opp": "opposite",
    "nr": "near",
    "w": "west",
    "e": "east",
    "n": "north",
    "s": "south",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "no": "number",
    "hno": "house number",
}

# Legal suffixes (tokens that appear at the end of business names)
# We track these to create a "core name" without legal qualifiers
LEGAL_SUFFIXES = {
    "corporation", "corp", "incorporated", "inc", "limited", "ltd",
    "private", "pvt", "company", "co", "llc", "llp", "lp", "pc",
    "liability", "partnership", "holdings", "holding", "enterprises",
    "enterprise", "solutions", "group", "associates",
    "sarl", "sasu", "sas", "eurl", "sa", "sci", "fils", "gmbh", "ag",
    "societe", "anonyme", "responsabilite", "limitee", "simplifiee",
    "actions", "unipersonnelle", "civile", "immobiliere",
}


# ─── Base Cleaning ────────────────────────────────────────────────────────────

def safe_str(text) -> str:
    """Safely convert any value (including NaN) to string."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text).strip()


def unicode_clean(text: str) -> str:
    """
    NFKD normalization: decompose accented characters into base + accent.
    Then remove the accent marks (combining characters).
    
    Example: 'Café' -> 'Cafe', 'naïve' -> 'naive'
    This helps match French business names that may or may not have accents.
    """
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def clean_base(text: str) -> str:
    """
    Basic cleaning pipeline:
    1. Unicode normalization (remove accents)
    2. Lowercase
    3. Replace & with 'and', + with 'plus'
    4. Remove URLs, emails
    5. Replace punctuation with spaces
    6. Collapse whitespace
    """
    if not text:
        return ""
    
    text = unicode_clean(text).lower()
    
    # Normalize separators
    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")
    
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bwww\.\S+", " ", text)
    text = re.sub(r"\.(com|org|net|in|fr|co|io|edu|gov)\b", " ", text)
    
    # Remove synthetic ID tags like (ID: 84923)
    text = re.sub(r"[\(\[\{]\s*id\s*:\s*\d+\s*[\)\]\}]", " ", text)
    
    # Replace all remaining punctuation with space
    # (but KEEP digits — important for house numbers, PIN codes)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Business Name Normalization ──────────────────────────────────────────────

def normalize_business_name(text) -> str:
    """
    Normalize a business name for comparison:
    1. Clean base text
    2. Expand abbreviations (corp -> corporation, pvt -> private, etc.)
    
    Returns a single normalized string.
    
    Example:
      "ABC Corp. Pvt. Ltd." -> "abc corporation private limited"
      "Tech Svc Inc"        -> "technology services incorporated"
    """
    text = safe_str(text)
    cleaned = clean_base(text)
    if not cleaned:
        return ""
    
    tokens = cleaned.split()
    expanded = [NAME_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_business_name_tokens(text) -> list:
    """Return normalized business name as a list of tokens (for Jaccard etc.)."""
    norm = normalize_business_name(text)
    return norm.split() if norm else []


def extract_core_name(text) -> str:
    """
    Extract the 'core' business name by removing legal suffixes.
    
    Example:
      "Sunrise Technologies Private Limited" -> "sunrise technology"
      (removes "private", "limited", expands "technologies" -> "technology")
    
    This helps compare just the meaningful part of the name.
    """
    norm = normalize_business_name(text)
    tokens = norm.split()
    # Remove legal suffix tokens from the end (and anywhere they appear)
    core = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not core:
        core = tokens  # Fallback: if only suffixes remain, keep all
    return " ".join(core)


# ─── Address Normalization ────────────────────────────────────────────────────

def normalize_address(text) -> str:
    """
    Normalize an address for comparison:
    1. Clean base text
    2. Expand common street abbreviations (Rd -> road, St -> street, etc.)
    
    Preserves:
    - House/building numbers (important for matching)
    - Postal/PIN codes (strong matching signal)
    - Street names and locality tokens
    
    Example:
      "17560 Ellis Rd, Tahlequah, OK" -> "17560 ellis road tahlequah ok"
    """
    text = safe_str(text)
    cleaned = clean_base(text)
    if not cleaned:
        return ""
    
    tokens = cleaned.split()
    expanded = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_address_tokens(text) -> list:
    """Return normalized address as a list of tokens."""
    norm = normalize_address(text)
    return norm.split() if norm else []


def extract_address_components(text) -> dict:
    """
    Parse an address into key components:
    - street_number: leading numeric token (house/building number)
    - postal_code: 5 or 6 digit standalone code (US ZIP or India PIN)
    - tokens: all other meaningful tokens
    - clean: full normalized address string
    
    Example:
      "1795 Westchester Dr, High Point, NC 27262"
      -> {street_number: "1795", postal_code: "27262",
          tokens: {"westchester", "drive", "high", "point", "nc"}}
    """
    norm = normalize_address(text)
    if not norm:
        return {"street_number": "", "postal_code": "", "tokens": set(), "clean": ""}
    
    tokens = norm.split()
    street_number = ""
    postal_code = ""
    other_tokens = set()
    
    for t in tokens:
        # Postal code: exactly 5 or 6 consecutive digits (US ZIP = 5, India PIN = 6)
        if re.fullmatch(r"\d{5,6}", t) and not postal_code:
            postal_code = t
        # House/street number: 1-6 digits, possibly with leading zeros
        elif re.fullmatch(r"0*\d{1,6}", t) and not street_number:
            stripped = t.lstrip("0") or "0"
            street_number = stripped
        else:
            other_tokens.add(t)
    
    return {
        "street_number": street_number,
        "postal_code": postal_code,
        "tokens": other_tokens,
        "clean": norm,
    }


# ─── Country Normalization ────────────────────────────────────────────────────

def normalize_country(text) -> str:
    """
    Normalize a country string.
    
    Important: We do NOT hard-code a whitelist of countries.
    The test data includes France which is not in training data.
    We simply lowercase and clean the string.
    
    Example:
      "US" -> "us"
      "India" -> "india"
      "France" -> "france"
    """
    text = safe_str(text)
    return clean_base(text)


# ─── Combined Text Representations ───────────────────────────────────────────

def build_combined_text(name, address, country) -> str:
    """
    Build a combined text representation for embedding or TF-IDF.
    Uses [SEP] tokens to separate fields so the model can distinguish them.
    
    Example:
      "abc restaurant private limited [SEP] 12 mg road coimbatore [SEP] india"
    """
    name_norm = normalize_business_name(name)
    addr_norm = normalize_address(address)
    ctry_norm = normalize_country(country)
    return f"{name_norm} [SEP] {addr_norm} [SEP] {ctry_norm}"


def build_name_only_text(name) -> str:
    """Normalized business name only (used for name-focused TF-IDF)."""
    return normalize_business_name(name)


def build_address_only_text(address) -> str:
    """Normalized address only (used for address-focused TF-IDF)."""
    return normalize_address(address)


# ─── Apply to DataFrame ───────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to a source DataFrame.
    
    New columns added:
    - business_name_normalized: cleaned + abbreviation-expanded name
    - business_address_normalized: cleaned + abbreviation-expanded address
    - country_normalized: cleaned country string
    - core_name: name without legal suffixes
    - combined_text: all fields joined with [SEP]
    - name_only_text: name for TF-IDF
    - address_only_text: address for TF-IDF
    """
    df = df.copy()
    
    print(f"    Normalizing business names...")
    df["business_name_normalized"] = df["business_name"].apply(normalize_business_name)
    df["core_name"] = df["business_name"].apply(extract_core_name)
    
    print(f"    Normalizing addresses...")
    df["business_address_normalized"] = df["business_address"].apply(normalize_address)
    
    print(f"    Normalizing countries...")
    df["country_normalized"] = df["country"].apply(normalize_country)
    
    print(f"    Building combined texts...")
    df["combined_text"] = df.apply(
        lambda r: build_combined_text(r["business_name"], r["business_address"], r["country"]),
        axis=1
    )
    df["name_only_text"]    = df["business_name_normalized"]
    df["address_only_text"] = df["business_address_normalized"]
    
    # Extract first token of name (useful for blocking)
    df["name_first_token"] = df["business_name_normalized"].apply(
        lambda x: x.split()[0] if x.strip() else ""
    )
    # Extract first 3 chars of name (prefix blocking)
    df["name_prefix3"] = df["business_name_normalized"].apply(
        lambda x: x[:3] if len(x) >= 3 else x
    )
    
    return df


if __name__ == "__main__":
    # Quick self-test
    tests = [
        ("ABC Corp. Pvt. Ltd.", "12 MG Rd, Coimbatore, TN 641001", "India"),
        ("Orelee's Barbershop", "1795 Westchester Dr, High Point, NC 27265", "US"),
        ("Tech Svc Inc", "105 Elm St, Morganton, NC", "US"),
        ("Cafe de Paris SARL", "175 Blvd Roosevelt, Bordeaux", "France"),
    ]
    print("Testing normalization:")
    for name, addr, country in tests:
        print(f"\n  Input:   name={name!r}  addr={addr!r}")
        print(f"  Name:    {normalize_business_name(name)!r}")
        print(f"  Core:    {extract_core_name(name)!r}")
        print(f"  Addr:    {normalize_address(addr)!r}")
        print(f"  Country: {normalize_country(country)!r}")
        comps = extract_address_components(addr)
        print(f"  Addr components: street_num={comps['street_number']!r}  "
              f"postal={comps['postal_code']!r}  tokens={comps['tokens']}")
