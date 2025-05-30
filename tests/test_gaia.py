```python
import unittest
import os
import logging
from unittest.mock import patch, Mock
from cryptography.fernet import Fernet
from src.utils.helpers import setup_logging, encrypt_data, decrypt_data
from src.core.gaia import VersionControl

class TestGaia(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = "test_versions"
        self.log_file = "logs/gaia_test.log"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        self.logger = setup_logging()
        self.key = Fernet.generate_key()
        self.version_control = VersionControl(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if os.path.exists(self.test_dir):
            for f in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, f))
            os.rmdir(self.test_dir)
        if os.path.exists("logs"):
            os.rmdir("logs")

    def test_setup_logging(self):
        """Test logging configuration."""
        self.logger.info("Test log message")
        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, "r") as f:
            log_content = f.read()
        self.assertIn("Test log message", log_content)

    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        data = "Secret message"
        encrypted = encrypt_data(data, self.key)
        self.assertIsNotNone(encrypted)
        decrypted = decrypt_data(encrypted, self.key)
        self.assertEqual(decrypted, data)

    def test_encrypt_invalid_key(self):
        """Test encryption with invalid key."""
        data = "Secret message"
        invalid_key = b"invalid_key"
        encrypted = encrypt_data(data, invalid_key)
        self.assertIsNone(encrypted)
        with open(self.log_file, "r") as f:
            log_content = f.read()
        self.assertIn("Encryption error", log_content)

    def test_version_control_save_load(self):
        """Test version control save and load."""
        data = {"code": "print('Hello')", "metadata": {"version": "v0"}}
        self.version_control.save_version(data, "v0")
        version_path = os.path.join(self.test_dir, "model_v0.pkl")
        self.assertTrue(os.path.exists(version_path))
        loaded_data = self.version_control.load_version("v0")
        self.assertEqual(loaded_data["code"], data["code"])

    @patch("websocket.WebSocketApp")
    def test_websocket_connection(self, mock_ws):
        """Test WebSocket connection (mocked)."""
        from src.core.gaia import Gaia  # Import here to avoid initialization
        mock_ws_instance = Mock()
        mock_ws.return_value = mock_ws_instance
        gaia = Gaia()
        gaia.start_websocket()
        mock_ws.assert_called_with(
            "ws://localhost:8766",
            on_message=gaia.on_message,
            on_error=gaia.on_error,
            on_close=gaia.on_close,
            on_open=gaia.on_open
        )
        mock_ws_instance.run_forever.assert_called()

if __name__ == "__main__":
    unittest.main()
```
