from web3 import Web3
from solcx import compile_standard, install_solc
import json
import time

# Install specific Solidity compiler version
print("⚙️ Installing Solidity Compiler...")
install_solc("0.8.0")

class BlockchainLogger:
    def __init__(self):
        # 1. Start a temporary local blockchain (Mock)
        self.w3 = Web3(Web3.EthereumTesterProvider())
        self.w3.eth.default_account = self.w3.eth.accounts[0]
        print(f"🔗 Blockchain Connected. Account: {self.w3.eth.default_account}")

        # 2. Compile the Smart Contract
        with open("AuditLog.sol", "r") as file:
            audit_file_content = file.read()

        compiled_sol = compile_standard({
            "language": "Solidity",
            "sources": {"AuditLog.sol": {"content": audit_file_content}},
            "settings": {"outputSelection": {"*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}}},
        }, solc_version="0.8.0")

        bytecode = compiled_sol["contracts"]["AuditLog.sol"]["OmniShieldAudit"]["evm"]["bytecode"]["object"]
        self.abi = compiled_sol["contracts"]["AuditLog.sol"]["OmniShieldAudit"]["abi"]

        # 3. Deploy the Contract
        AuditContract = self.w3.eth.contract(abi=self.abi, bytecode=bytecode)
        tx_hash = AuditContract.constructor().transact()
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        self.contract = self.w3.eth.contract(address=tx_receipt.contractAddress, abi=self.abi)
        print(f"📜 Smart Contract Deployed at: {tx_receipt.contractAddress}")

    def log_event(self, action_type, data_hash):
        """Writes an immutable log to the blockchain"""
        tx_hash = self.contract.functions.addLog(action_type, data_hash).transact()
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        count = self.contract.functions.getLogCount().call()
        print(f"⛓️ BLOCKCHAIN LOG: Action '{action_type}' recorded. Total Logs: {count}")

# Simple test if run directly
if __name__ == "__main__":
    logger = BlockchainLogger()
    logger.log_event("TEST_ACTION", "hash_123")