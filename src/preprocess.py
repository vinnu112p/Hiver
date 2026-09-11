"""
Phase 5: Data Cleaning & Normalization
Implements principled, justified text preprocessing for customer support tweets.
Preserves critical semantic signals (negation, casing, technical terms) while stripping Twitter routing noise.
"""
import re
import html

# Regex patterns
RE_HTML_TAGS = re.compile(r"<[^>]+>")
RE_MENTIONS_START = re.compile(r"^(@[A-Za-z0-9_]+\s*)+")
RE_INLINE_MENTIONS = re.compile(r"@[A-Za-z0-9_]+")
RE_WHITESPACE = re.compile(r"\s+")
RE_URLS = re.compile(r"https?://\S+|www\.\S+")

def clean_customer_text(text: str, remove_inline_mentions: bool = True) -> str:
    """
    Cleans customer message while strictly preserving negations and problem descriptions.
    - Unescapes HTML entities (&amp; -> &)
    - Strips leading @handle routing tags
    - Optionally replaces inline mentions with @user to avoid leaking IDs
    - Normalizes irregular whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unescape HTML entities
    cleaned = html.unescape(text)

    # 2. Strip HTML tags if any
    cleaned = RE_HTML_TAGS.sub(" ", cleaned)

    # 3. Strip leading handle mentions (e.g. "@SpotifyCares @115821 hello" -> "hello")
    cleaned = RE_MENTIONS_START.sub("", cleaned).strip()

    # 4. Normalize inline mentions if requested
    if remove_inline_mentions:
        cleaned = RE_INLINE_MENTIONS.sub("@user", cleaned)

    # 5. Normalize whitespace
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip()

    return cleaned

def clean_support_text(text: str) -> str:
    """
    Cleans support response while preserving official guidance, links, and tone.
    - Unescapes HTML entities
    - Strips leading @customer mentions
    - Preserves troubleshooting URLs, steps, and support signatures (e.g. /CK)
    - Normalizes whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = html.unescape(text)
    cleaned = RE_HTML_TAGS.sub(" ", cleaned)
    cleaned = RE_MENTIONS_START.sub("", cleaned).strip()
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip()

    return cleaned

def is_valid_interaction(customer_text: str, support_text: str, min_words: int = 3) -> tuple[bool, str]:
    """
    Quality gatekeeper to discard noisy or malformed interactions.
    Returns (is_valid, rejection_reason).
    """
    c_clean = clean_customer_text(customer_text)
    s_clean = clean_support_text(support_text)

    if not c_clean:
        return False, "Empty customer message"
    if not s_clean:
        return False, "Empty support reply"

    c_words = c_clean.split()
    if len(c_words) < min_words:
        return False, f"Customer message too short ({len(c_words)} words < {min_words})"

    s_words = s_clean.split()
    if len(s_words) < 2:
        return False, "Support response too short"

    # Reject messages that are solely punctuation or emojis
    if not re.search(r"[a-zA-Z0-9]", c_clean):
        return False, "Customer message lacks alphanumeric characters"

    return True, "Valid"
