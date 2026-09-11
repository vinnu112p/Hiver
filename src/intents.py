"""
Phase 6: Intent Definitions & Metadata
Defines the empirical 7-intent taxonomy for Spotify Customer Support.
Derived directly from frequency analysis of 41,272 authentic customer tweets.
"""

INTENT_TAXONOMY = {
    "billing_subscription": {
        "description": "Questions or issues regarding payment methods, recurring charges, subscription tiers (Premium, Family, Student), and pricing.",
        "risk_level": "medium",
        "default_action": "auto_handle",
        "keywords": ["charge", "charged", "billing", "bill", "subscription", "premium", "student", "discount", "family", "payment", "card"]
    },
    "cancellation_refund": {
        "description": "Requests to cancel active subscriptions, stop future recurring billing, or receive monetary refunds for unwanted charges.",
        "risk_level": "high",
        "default_action": "escalate",
        "keywords": ["cancel", "cancelled", "cancelling", "refund", "money back", "unsubscribe", "stop charging"]
    },
    "account_access_security": {
        "description": "Trouble logging in, password resets, compromised or hacked accounts, email address changes, or unrecognized account activity.",
        "risk_level": "critical",
        "default_action": "escalate",
        "keywords": ["login", "log in", "password", "reset", "email", "hacked", "stolen", "access", "account", "compromised"]
    },
    "audio_playback_issue": {
        "description": "Music pausing unexpectedly, song skipping, offline download playback failures, sound distortion, or Bluetooth/speaker streaming bugs.",
        "risk_level": "low",
        "default_action": "auto_handle",
        "keywords": ["play", "playing", "pause", "stops", "skipping", "song", "track", "music", "sound", "offline", "download", "bluetooth"]
    },
    "app_crash_technical": {
        "description": "Application crashing on launch, freezing, UI bugs, black screens, or failed updates across desktop, iOS, or Android apps.",
        "risk_level": "low",
        "default_action": "auto_handle",
        "keywords": ["crash", "crashing", "update", "freeze", "freezing", "black screen", "error", "bug", "glitch", "reinstall"]
    },
    "library_playlist_content": {
        "description": "Missing playlists, deleted saved tracks, greyed-out songs, local audio file sync issues, or podcast display problems.",
        "risk_level": "low",
        "default_action": "auto_handle",
        "keywords": ["playlist", "library", "album", "artist", "songs missing", "deleted", "local files", "disappeared", "greyed out", "sync"]
    },
    "other_support": {
        "description": "Out-of-scope queries, compliments, feature requests, marketing campaigns, or vague messages that cannot be safely diagnosed.",
        "risk_level": "medium",
        "default_action": "escalate",
        "keywords": []
    }
}

ALL_INTENTS = list(INTENT_TAXONOMY.keys())

def get_intent_risk(intent: str) -> str:
    return INTENT_TAXONOMY.get(intent, {}).get("risk_level", "medium")

def get_intent_default_action(intent: str) -> str:
    return INTENT_TAXONOMY.get(intent, {}).get("default_action", "escalate")
