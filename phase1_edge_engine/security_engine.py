import os
import json
import hashlib
from cryptography.fernet import Fernet

# --- CONFIGURATION ---
RECOVERY_DB_PATH = "downloads/recovery_registry.json"
# In a real enterprise deployment, this key would be retrieved from a Vault/KMS
# We use a persistent key for this session to ensure decryption works after restarts
STATIC_KEY = b'G6Y-7_V8J_k5L-9_Z3X_c2V_b1N_m0A_s9D_f8G_h7J='
cipher_suite = Fernet(STATIC_KEY)

# Ensure database exists
if not os.path.exists("downloads"):
    os.makedirs("downloads")

if not os.path.exists(RECOVERY_DB_PATH):
    with open(RECOVERY_DB_PATH, "w") as f:
        json.dump({}, f)

def get_file_hash(data_bytes: bytes) -> str:
    """Generates a SHA-256 hash of the file contents."""
    return hashlib.sha256(data_bytes).hexdigest()

def encrypt_vault(data_bytes: bytes) -> bytes:
    """Encrypts raw data using the Enterprise session key."""
    return cipher_suite.encrypt(data_bytes)

def decrypt_vault(encrypted_bytes: bytes) -> bytes:
    """Decrypts data using the Enterprise session key."""
    return cipher_suite.decrypt(encrypted_bytes)

def register_access(file_hash: str, address: str):
    """
    Registers an authorized address (Owner or Guardian) in the recovery registry.
    Acts as the off-chain anchor for the ERC-4337 social recovery logic.
    """
    with open(RECOVERY_DB_PATH, "r") as f:
        db = json.load(f)
    
    if file_hash not in db:
        db[file_hash] = []
    
    clean_address = address.lower().strip()
    if clean_address not in db[file_hash]:
        db[file_hash].append(clean_address)
        
    with open(RECOVERY_DB_PATH, "w") as f:
        json.dump(db, f)

def check_authorization(file_hash: str, wallet_address: str) -> bool:
    """Checks if a wallet is authorized to decrypt the specific file hash."""
    with open(RECOVERY_DB_PATH, "r") as f:
        db = json.load(f)
    
    authorized_list = db.get(file_hash, [])
    return wallet_address.lower().strip() in authorized_list

def generate_zk_proof(file_hash: str, owner: str):
    """
    Simulates the generation of a ZK-SNARK proof.
    In production, this would interface with a circuit (Circom/ZoKrates).
    """
    return {
        "proof_type": "ZK-SNARK",
        "file_hash": file_hash,
        "owner_commitment": hashlib.sha256(owner.encode()).hexdigest(),
        "timestamp": int(os.path.getmtime(RECOVERY_DB_PATH)),
        "status": "verified"
    }