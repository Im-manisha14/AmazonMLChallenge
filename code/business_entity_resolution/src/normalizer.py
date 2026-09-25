"""
Normalization module for Business Entity Resolution.
Handles unicode NFKD normalization, punctuation stripping, abbreviation expansion,
and address component parsing (street number, street name, city, postal code).
"""

import re
import unicodedata

# Common abbreviations mapping
NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "sarl": "societe a responsabilite limitee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "sas": "societe par actions simplifiee",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "sa": "societe anonyme",
    "sci": "societe civile immobiliere",
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
    "dist": "district",
    "engr": "engineering",
    "ent": "enterprises",
    "ind": "industries",
}

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
    "unit": "unit",
    "bldg": "building",
    "opp": "opposite",
    "b/h": "behind",
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
    "h no": "house number",
    "hno": "house number",
}

# Legal suffixes to strip when creating "core" name
LEGAL_SUFFIXES = {
    "corporation", "corp", "incorporated", "inc", "limited", "ltd",
    "private", "pvt", "company", "co", "llc", "llp", "lp", "pc",
    "liability", "partnership", "holdings", "holding", "enterprises",
    "enterprise", "solutions", "group", "associates",
    "sarl", "sasu", "sas", "eurl", "sa", "sci", "fils", "gmbh", "ag"
}


def unicode_clean(text: str) -> str:
    """NFKD unicode normalization to decompose accents, ligatures, etc."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    # Remove accent marks (combining characters)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def clean_text_base(text: str) -> str:
    """Basic cleaning: lowercase, remove URLs, strip noisy punctuation, normalize spaces."""
    if not text:
        return ""
    text = unicode_clean(text).lower()
    
    # Handle & and +
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\+", " plus ", text)
    
    # Remove URLs, domains, www, email
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bwww\.\S+", " ", text)
    text = re.sub(r"\.(com|org|net|in|fr|co|io|edu|gov)\b", " ", text)
    
    # Remove synthetic ID tags e.g. (ID: 84923) or [ID: 123]
    text = re.sub(r"[\(\[\{]\s*id\s*:\s*\d+\s*[\)\]\}]", " ", text)
    
    # Remove brackets, quotes, dashes, punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_text(text: str, is_address: bool = False) -> str:
    """Expand abbreviations in text using the domain dictionary."""
    cleaned = clean_text_base(text)
    if not cleaned:
        return ""
    
    mapping = ADDRESS_ABBREVIATIONS if is_address else NAME_ABBREVIATIONS
    tokens = cleaned.split()
    expanded_tokens = [mapping.get(t, t) for t in tokens]
    return " ".join(expanded_tokens)


def extract_core_name(name: str) -> str:
    """Extract business name without legal suffixes or noisy qualifiers."""
    cleaned = clean_text_base(name)
    tokens = cleaned.split()
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and t not in {"services", "service", "dba"}]
    if not core_tokens:
        core_tokens = tokens
    return " ".join(core_tokens)


def extract_address_components(address: str) -> dict:
    """
    Parse an address into components:
    - street_number: normalized digits (stripping leading zeros, e.g. '0017560' -> '17560')
    - postal_code: 5 or 6 digit postal code if found
    - street_tokens: remaining tokens
    - clean: full expanded address
    """
    clean_addr = expand_text(address, is_address=True)
    if not clean_addr:
        return {
            "street_number": "",
            "postal_code": "",
            "tokens": set(),
            "clean": ""
        }
    
    tokens = clean_addr.split()
    street_number = ""
    postal_code = ""
    other_tokens = set()
    
    for t in tokens:
        # Check for postal code: 5 or 6 consecutive digits
        # (postal codes are usually at the end of the address or standalone 5-6 digits)
        num_clean = t.lstrip("0")
        if re.fullmatch(r"\d{5,6}", t) and not postal_code:
            postal_code = t
        elif re.fullmatch(r"0*\d{1,6}", t) and not street_number:
            street_number = num_clean if num_clean else "0"
        else:
            other_tokens.add(t)
            
    return {
        "street_number": street_number,
        "postal_code": postal_code,
        "tokens": other_tokens,
        "clean": clean_addr
    }

