from web3 import Web3
from web3.exceptions import Web3Exception
from solcx import compile_standard, install_solc
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("omni_shield")

_SOL_FILE    = Path(__file__).parent / "AuditLog.sol"
_LEDGER_FILE = Path(__file__).parent / "ledger.json"

# ── Network configuration ─────────────────────────────────────────────────────
# All values are read lazily inside BlockchainLogger.__init__ so that
# systemd EnvironmentFile= entries are visible at instantiation time
# (not at module-import time, which happens before the env is fully loaded).
_GANACHE_URL = "http://127.0.0.1:7545"

try:
    install_solc("0.8.0")
except Exception as _solc_err:
    log.warning(f"[BLOCKCHAIN] install_solc failed (will retry at first use): {_solc_err}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_connection(w3: Web3, rpc_url: str) -> None:
    """
    Verify that the configured RPC endpoint is reachable.
    Raises ConnectionError instead of calling sys.exit() so the caller
    can decide whether the failure is fatal.
    """
    connected = w3.is_connected()
    log.info(f"[BLOCKCHAIN] Connected to {rpc_url}: {connected}")
    if not connected:
        raise ConnectionError(f"RPC endpoint not reachable: {rpc_url}")
    try:
        w3.eth.block_number
    except Exception as e:
        raise ConnectionError(f"RPC reachable but eth_blockNumber failed: {e}")


def _compile_contract() -> tuple[list, str]:
    source = _SOL_FILE.read_text()
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"AuditLog.sol": {"content": source}},
            "settings": {
                "outputSelection": {
                    "*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}
                }
            },
        },
        solc_version="0.8.0",
    )
    contract_data = compiled["contracts"]["AuditLog.sol"]["OmniShieldAudit"]
    abi      = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]
    return abi, bytecode


def _load_ledger() -> dict:
    if _LEDGER_FILE.exists():
        try:
            return json.loads(_LEDGER_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_ledger(data: dict) -> None:
    _LEDGER_FILE.write_text(json.dumps(data, indent=2))


# ── Core class ─────────────────────────────────────────────────────────────────

class BlockchainLogger:
    def __init__(self):
        # Read env vars here (lazy) — systemd EnvironmentFile= is loaded
        # before the service forks the Python process, but module-level
        # reads happen at import time which may precede env injection.
        rpc_url          = os.environ.get("BLOCKCHAIN_RPC_URL", _GANACHE_URL)
        private_key      = os.environ.get(
            "BLOCKCHAIN_PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
        contract_address = os.environ.get("BLOCKCHAIN_CONTRACT_ADDRESS", None)

        log.info(f"[BLOCKCHAIN] RPC: {rpc_url}")
        log.info(f"[BLOCKCHAIN] Contract env: {contract_address!r}")

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        _check_connection(self.w3, rpc_url)

        # Use private key account for production networks; fall back to
        # Ganache unlocked accounts for local dev.
        _is_ganache = "127.0.0.1" in rpc_url or "localhost" in rpc_url
        if _is_ganache:
            self.w3.eth.default_account = self.w3.eth.accounts[0]
            self._account   = self.w3.eth.default_account
            self._sign_mode = "ganache"
            self._eth_account = None
        else:
            from eth_account import Account as _Account
            self._eth_account = _Account.from_key(private_key)
            self._account      = self._eth_account.address
            self.w3.eth.default_account = self._account
            self._sign_mode = "key"

        log.info(f"[BLOCKCHAIN] Wallet: {self._account}")
        log.info(f"[BLOCKCHAIN] Sign mode: {self._sign_mode}")

        self.abi, bytecode = _compile_contract()

        # ABI version stamp — bump this string whenever addRecord signature changes
        _ABI_VERSION = "v4-no-claim"

        # Priority: env var > ledger.json > deploy fresh
        ledger          = _load_ledger()
        cached_address  = contract_address or ledger.get("contract_address")
        contract_loaded = False

        if cached_address:
            try:
                cached_address = self.w3.to_checksum_address(cached_address)
            except Exception:
                pass
            log.info(f"[BLOCKCHAIN] Trying contract at: {cached_address}")
            try:
                code = self.w3.eth.get_code(cached_address)
                if code and code not in (b"", b"0x"):
                    self.contract = self.w3.eth.contract(
                        address=cached_address, abi=self.abi
                    )
                    log.info(f"[BLOCKCHAIN] Loaded contract at: {cached_address}")
                    contract_loaded = True
                else:
                    log.warning(f"[BLOCKCHAIN] No bytecode at {cached_address} — redeploying")
            except Exception as exc:
                log.warning(f"[BLOCKCHAIN] Contract address unusable ({exc}) — redeploying")

        if not contract_loaded:
            if _is_ganache:
                log.info("[BLOCKCHAIN] Deploying new contract on Ganache …")
                AuditContract = self.w3.eth.contract(abi=self.abi, bytecode=bytecode)
                tx_receipt    = self._send_transaction(
                    AuditContract.constructor(), gas=1_000_000
                )
                self.contract = self.w3.eth.contract(
                    address=tx_receipt.contractAddress, abi=self.abi
                )
                log.info(f"[BLOCKCHAIN] Deployed at: {tx_receipt.contractAddress}")
                ledger["contract_address"] = tx_receipt.contractAddress
                ledger["abi_version"]      = _ABI_VERSION
                _save_ledger(ledger)
            else:
                raise RuntimeError(
                    "BLOCKCHAIN_CONTRACT_ADDRESS is not set (or has no bytecode). "
                    "Deploy the contract first and set the env var."
                )

    # ── Transaction helper ─────────────────────────────────────────────────────

    def _send_transaction(self, fn_or_constructor, gas: int = 200_000):
        """Submit a contract call / constructor and return the receipt.

        Ganache mode: uses unlocked default_account via transact().
        Production mode: signs the transaction with _PRIVATE_KEY before
        broadcasting (required for Polygon Amoy / mainnet).
        """
        if self._sign_mode == "ganache":
            tx_hash = fn_or_constructor.transact()
            return self.w3.eth.wait_for_transaction_receipt(tx_hash)

        # Production: build → sign → send
        nonce = self.w3.eth.get_transaction_count(self._account)
        tx = fn_or_constructor.build_transaction({
            "from":     self._account,
            "gas":      gas,
            "nonce":    nonce,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self._eth_account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def add_record(
        self, doc_hash_bytes: bytes, redaction_count: int, doc_type: str,
        owner_address: str = "",
    ) -> None:
        """Call addRecord(bytes32, uint256, string, address) on the deployed contract.

        owner_address: optional MetaMask / user wallet to attribute the record to.
        The transaction is always sent from the Ganache default account (onlyOwner),
        but the on-chain recordOwner and _userRecords index will use owner_address
        when it is a valid 42-char hex Ethereum address.
        """
        if owner_address and len(owner_address) == 42 and owner_address.startswith('0x'):
            try:
                record_owner = self.w3.to_checksum_address(owner_address)
            except Exception:
                record_owner = "0x0000000000000000000000000000000000000000"
        else:
            record_owner = "0x0000000000000000000000000000000000000000"

        log.info(
            f"  [BLOCKCHAIN] add_record: raw owner_address={owner_address!r} "
            f"→ record_owner={record_owner}"
        )
        self._send_transaction(
            self.contract.functions.addRecord(
                doc_hash_bytes, redaction_count, doc_type, record_owner
            )
        )
        log.info(
            f"  [BLOCKCHAIN] addRecord confirmed — "
            f"hash={doc_hash_bytes.hex()[:16]}… count={redaction_count} "
            f"owner={record_owner[:10]}…"
        )

    def verify(self, doc_hash_bytes: bytes) -> dict:
        """Call verify(bytes32) and return a plain dict."""
        exists, redaction_count, timestamp = (
            self.contract.functions.verify(doc_hash_bytes).call()
        )
        return {
            "exists": exists,
            "redaction_count": redaction_count,
            "timestamp": timestamp,
        }

    def get_records_for_user(self, user_address: str) -> list[dict]:
        """
        Call getRecordsByUser(user_address) — queries the _userRecords index that
        is populated when addRecord() is called with a non-zero owner_address.
        Falls back to getRecordsByOwner if the user index is empty (e.g. when
        user_address happens to be the Ganache default account).
        """
        checksum_addr = self.w3.to_checksum_address(user_address)
        raw_hashes: list[bytes] = (
            self.contract.functions.getRecordsByUser(checksum_addr).call()
        )
        if not raw_hashes:
            raw_hashes = self.contract.functions.getRecordsByOwner(checksum_addr).call()

        records = []
        for h in raw_hashes:
            info = self.verify(h)
            records.append({
                "sha256_hex":      h.hex(),
                "redaction_count": info["redaction_count"],
                "timestamp":       info["timestamp"],
            })
        log.info(f"  [BLOCKCHAIN] Records for {user_address[:10]}…: {len(records)} total")
        return records

    def get_my_records(self) -> list[dict]:
        """
        Call getRecordsByOwner(account), then enrich each hash with verify().
        Prints a formatted table and returns the list of dicts.
        """
        account = self.w3.eth.default_account
        raw_hashes: list[bytes] = (
            self.contract.functions.getRecordsByOwner(account).call()
        )

        records = []
        for h in raw_hashes:
            info = self.verify(h)
            records.append(
                {
                    "sha256_hex":      h.hex(),
                    "redaction_count": info["redaction_count"],
                    "timestamp":       info["timestamp"],
                }
            )

        print(f"\n📋 Records anchored by {account}  ({len(records)} total)")
        if records:
            print(f"  {'SHA-256':<64}  {'Redactions':>10}  Timestamp")
            print("  " + "-" * 84)
            for rec in records:
                print(
                    f"  {rec['sha256_hex']}  "
                    f"{rec['redaction_count']:>10}  "
                    f"{rec['timestamp']}"
                )
        return records


# ── Module-level singleton ─────────────────────────────────────────────────────

_logger_instance: BlockchainLogger | None = None


def _get_blockchain_logger() -> "BlockchainLogger | None":
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance
    try:
        _logger_instance = BlockchainLogger()
        return _logger_instance
    except Exception as e:
        log.warning(f"Blockchain unavailable: {e}")
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def anchor_audit(audit_json_path: str, owner_address: str = "") -> None:
    """
    Read a RedactionAgent .audit.json sidecar, compute SHA-256 of the original
    source file, and anchor it as a bytes32 hash + redaction_count to the contract.

    owner_address: optional MetaMask wallet to attribute the record to.
    Non-fatal: any failure is logged as a warning so the pipeline continues.
    """
    log.info(f"  [BLOCKCHAIN] anchor_audit called: owner_address={owner_address!r}")
    logger = _get_blockchain_logger()
    if logger is None:
        log.warning("Blockchain unavailable — skipping anchor")
        return
    try:
        audit           = json.loads(Path(audit_json_path).read_text())
        source_path     = audit.get("source", "")
        redaction_count = int(audit.get("redaction_count", 0))
        doc_type        = audit.get("doc_type", "unknown")

        source_bytes    = Path(source_path).read_bytes()
        sha256_hex      = hashlib.sha256(source_bytes).hexdigest()
        doc_hash_bytes  = bytes.fromhex(sha256_hex)   # 32 bytes → bytes32

        log.info(
            f"  [BLOCKCHAIN] anchor_audit → add_record: "
            f"hash={sha256_hex[:16]}… owner={owner_address!r}"
        )
        logger.add_record(doc_hash_bytes, redaction_count, doc_type, owner_address)
        log.info(
            f"  [BLOCKCHAIN] Anchored sha256={sha256_hex[:16]}… count={redaction_count}"
        )
    except Exception as exc:
        exc_str = str(exc)
        if "Record already exists" in exc_str or "already exists" in exc_str.lower():
            log.info(f"  [BLOCKCHAIN] anchor_audit: already anchored, skipping.")
        else:
            log.warning(f"  [BLOCKCHAIN] anchor_audit failed (non-fatal): {exc}")


def anchor_record(
    sha256_hex: str, redaction_count: int, doc_type: str = "unknown",
    owner_address: str = "", salt: str = None,
) -> tuple[str, str]:
    """
    Anchor a document directly by SHA-256 hex string.

    To prevent linkability (identical documents producing the same on-chain
    hash), the value stored on-chain is a salted SHA-256:

        anchor_hash = SHA256(sha256_hex + salt)

    where salt defaults to "{owner_address}:{unix_timestamp}".  The original
    sha256_hex is preserved in audit JSON so users can still verify by file.

    Returns: (anchor_hash_hex, salt) so the caller can record both.
    """
    import time as _time
    if salt is None:
        salt = f"{owner_address}:{int(_time.time())}"

    anchor_hash_hex = hashlib.sha256((sha256_hex + salt).encode()).hexdigest()

    log.info(
        f"  [BLOCKCHAIN] anchor_record: "
        f"original={sha256_hex[:16]}… salt={salt[:24]}… "
        f"anchor={anchor_hash_hex[:16]}… owner={owner_address!r} doc_type={doc_type!r}"
    )
    logger = _get_blockchain_logger()
    if logger is None:
        log.warning("Blockchain unavailable — skipping anchor")
        return anchor_hash_hex, salt
    try:
        doc_hash_bytes = bytes.fromhex(anchor_hash_hex)
        logger.add_record(doc_hash_bytes, redaction_count, doc_type, owner_address)
        log.info(
            f"  [BLOCKCHAIN] Anchored anchor_hash={anchor_hash_hex[:16]}… "
            f"count={redaction_count}"
        )
    except Exception as exc:
        exc_str = str(exc)
        if "Record already exists" in exc_str or "already exists" in exc_str.lower():
            log.info(f"  [BLOCKCHAIN] Already anchored, skipping: {anchor_hash_hex[:16]}…")
        else:
            log.warning(f"  [BLOCKCHAIN] anchor_record failed (non-fatal): {exc}")
    return anchor_hash_hex, salt


def verify_document(file_path: str) -> dict:
    """
    Compute SHA-256 of *file_path* and query the contract's verify() function.

    Returns a dict: exists, redaction_count, timestamp, sha256_hex.
    Prints a human-readable result to stdout.
    """
    try:
        file_bytes     = Path(file_path).read_bytes()
        sha256_hex     = hashlib.sha256(file_bytes).hexdigest()
        doc_hash_bytes = bytes.fromhex(sha256_hex)

        result                = _get_blockchain_logger().verify(doc_hash_bytes)
        result["sha256_hex"]  = sha256_hex

        if result["exists"]:
            print(
                f"✅ Document found in ledger — "
                f"redactions={result['redaction_count']}, ts={result['timestamp']}"
            )
        else:
            print(f"❌ Document not found in ledger (sha256={sha256_hex[:16]}…)")

        return result
    except Exception as exc:
        log.warning(f"  [BLOCKCHAIN] verify_document failed: {exc}")
        return {
            "exists": False,
            "redaction_count": 0,
            "timestamp": 0,
            "sha256_hex": "",
            "error": str(exc),
        }


def get_my_records() -> list[dict]:
    """
    Return all document hashes anchored by the current wallet, enriched with
    redaction counts from verify().  Also prints a formatted table to stdout.
    """
    return _get_blockchain_logger().get_my_records()


def get_records_for_user(wallet_address: str) -> list[dict]:
    """
    Return all records where recordOwner == wallet_address (the _userRecords index).
    Falls back to getRecordsByOwner if the user index is empty.
    Non-fatal: returns empty list on any failure.
    """
    try:
        return _get_blockchain_logger().get_records_for_user(wallet_address)
    except Exception as exc:
        log.warning(f"  [BLOCKCHAIN] get_records_for_user failed: {exc}")
        return []


def get_records_for_wallet(wallet_address: str) -> list[dict]:
    """
    Return all document hashes anchored by a specific wallet address.
    Non-fatal: returns empty list on any failure.
    """
    try:
        logger = _get_blockchain_logger()
        checksum_addr = logger.w3.to_checksum_address(wallet_address)
        raw_hashes: list[bytes] = (
            logger.contract.functions.getRecordsByOwner(checksum_addr).call()
        )
        records = []
        for h in raw_hashes:
            info = logger.verify(h)
            records.append({
                "sha256_hex":      h.hex(),
                "redaction_count": info["redaction_count"],
                "timestamp":       info["timestamp"],
            })
        log.info(f"  [BLOCKCHAIN] Records for {wallet_address[:10]}…: {len(records)} total")
        return records
    except Exception as exc:
        log.warning(f"  [BLOCKCHAIN] get_records_for_wallet failed: {exc}")
        return []


# ── Fire-and-forget helpers ────────────────────────────────────────────────────

def _anchor_in_background(func, *args, **kwargs) -> None:
    """Run a blockchain anchor call in a daemon thread (fire-and-forget)."""
    def _run():
        try:
            func(*args, **kwargs)
        except Exception as exc:
            log.warning(f"[BLOCKCHAIN] Background anchor failed: {exc}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def anchor_record_async(
    sha256_hex: str, redaction_count: int, doc_type: str = "unknown",
    owner_address: str = "", salt: str = None,
) -> None:
    """Non-blocking anchor_record — returns immediately; anchor runs in background.

    salt should be computed by the caller before the async call so it can be
    stored in the audit JSON synchronously (the background thread may run later).
    """
    _anchor_in_background(anchor_record, sha256_hex, redaction_count, doc_type, owner_address, salt)


def anchor_audit_async(audit_json_path: str, owner_address: str = "") -> None:
    """Non-blocking anchor_audit — returns immediately; anchor runs in background."""
    _anchor_in_background(anchor_audit, audit_json_path, owner_address)


def verify_document_hash(sha256_hex: str) -> dict:
    """
    Query the contract's verify() function for an already-computed SHA-256 hex string.

    Returns a dict: exists, redaction_count, timestamp, sha256_hex.
    """
    try:
        doc_hash_bytes        = bytes.fromhex(sha256_hex)
        result                = _get_blockchain_logger().verify(doc_hash_bytes)
        result["sha256_hex"]  = sha256_hex
        return result
    except Exception as exc:
        log.warning(f"  [BLOCKCHAIN] verify_document_hash failed: {exc}")
        return {
            "exists": False,
            "redaction_count": 0,
            "timestamp": 0,
            "sha256_hex": sha256_hex,
            "error": str(exc),
        }


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = BlockchainLogger()

    h1 = bytes.fromhex(hashlib.sha256(b"document one").hexdigest())
    h2 = bytes.fromhex(hashlib.sha256(b"document two").hexdigest())
    logger.add_record(h1, 3, "id_card")
    logger.add_record(h2, 7, "form")

    print(logger.verify(h1))
    print(logger.verify(h2))
    logger.get_my_records()
