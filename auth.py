"""Local accounts with versioned, individually salted password digests."""
import hashlib
import hmac
import secrets
from datetime import date

ROUNDS = 600_000


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ROUNDS)
    return f"pbkdf2_sha256${ROUNDS}${salt.hex()}${digest.hex()}"


def register(db, sid, name, password, confirmation, daily_limit=3.0, today=None):
    if not isinstance(sid, str) or not 1 <= len(sid.strip()) <= 64:
        raise ValueError("Enter a student ID of 1–64 characters.")
    if not 1 <= len(name.strip()) <= 100:
        raise ValueError("Enter a full name of 1–100 characters.")
    if password != confirmation:
        raise ValueError("The passwords must match.")
    if not 8 <= len(password) <= 256:
        raise ValueError("Use a password of 8–256 characters.")
    db.create_account(sid.strip(), name.strip(), hash_password(password), daily_limit, today or date.today())


def authenticate(db, sid, password):
    encoded = db.password_hash(sid.strip())
    if not encoded or len(password) > 256:
        return False
    method, rounds, salt, expected = encoded.split("$")
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
    return method == "pbkdf2_sha256" and hmac.compare_digest(actual, bytes.fromhex(expected))


def logout(state):
    state.clear()
