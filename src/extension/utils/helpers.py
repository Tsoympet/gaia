from cryptography.fernet import Fernet
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/gaia.log",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def encrypt_data(data, key):
    cipher = Fernet(key)
    return cipher.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_data).decode()
