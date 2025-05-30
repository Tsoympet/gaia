```python
import logging
from cryptography.fernet import Fernet

def setup_logging():
    """
    Configure logging for the G.A.I.A application.
    Logs are written to 'logs/gaia.log' with INFO level.
    """
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/gaia.log",
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger

def encrypt_data(data, key):
    """
    Encrypts data using a Fernet key.
    
    Args:
        data (str): Data to encrypt
    key (bytes): Fernet key for encryption
    
    Returns:
        bytes: Encrypted data
    """
    try:
        cipher = Fernet(key)
        return cipher.encrypt(data.encode())
    except Exception as e:
        logging.error(f"Encryption error: {e}")
        return None

def decrypt_data(encrypted_data, key):
    """
    Decrypts data using a Fernet key.
    
    Args:
        encrypted_data (bytes): Data to decrypt
    key (bytes): Fernet key for decryption
    
    Returns:
        str: Decrypted data, or None if decryption fails
    """
    try:
        cipher = Fernet(key)
        return cipher.decrypt(encrypted_data).decode()
    except Exception as e:
        logging.error(f"Decryption error: {e}")
        return None
```
