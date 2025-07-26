"""
HEXTRA-API Secure Logger Utility
Environment-based logging for production security
"""
import os
import logging
from typing import Any

# Environment detection for production security
IS_DEV = os.getenv('ENV', 'development').lower() in ['development', 'dev', 'local']

class SecureLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if IS_DEV else logging.WARNING)

    def info(self, message: str, *args: Any):
        if IS_DEV:
            self.logger.info(message, *args)

    def warning(self, message: str, *args: Any):
        if IS_DEV:
            self.logger.warning(message, *args)

    def error(self, message: str, *args: Any):
        if IS_DEV:
            self.logger.error(message, *args)

    def debug(self, message: str, *args: Any):
        if IS_DEV:
            self.logger.debug(message, *args)

    def api(self, method: str, endpoint: str, status: int, message: str = ""):
        if IS_DEV:
            self.logger.info(f"🌐 API {method} {endpoint} -> {status} {message}")

    def sacred38(self, step: str, info: str):
        if IS_DEV:
            self.logger.info(f"✨ Sacred38: {step} - {info}")

    def hextra(self, component: str, action: str, details: str = ""):
        if IS_DEV:
            self.logger.info(f"🎯 HEXTRA {component}: {action} {details}")

logger = SecureLogger()