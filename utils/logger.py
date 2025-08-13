import os
import logging
from datetime import datetime as dt

# ==== Setup log paths ====
LOG_FILE = f"{dt.now().strftime('%Y_%m_%d_%I_%M_%S')}.log"
import os
import logging
from datetime import datetime as dt

# ==== Setup log paths ====
LOG_FILE = f"{dt.now().strftime('%Y_%m_%d_%I_%M_%S')}.log"
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# ==== Color Formatter for terminal ====
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"

# ==== Log format ====
log_format = "[ %(asctime)s ] %(lineno)d  %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)
color_formatter = ColorFormatter(log_format)

# ==== Set up logger ====
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler (plain)
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler.setFormatter(formatter)

# Terminal handler (colored)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(color_formatter)

# Add both handlers
logger.handlers = [file_handler, stream_handler]