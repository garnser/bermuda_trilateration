# logger.py
import logging
from config import LOGLEVEL

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # If 'training' is missing, set a default empty string.
        if 'training' not in record.__dict__:
            record.__dict__['training'] = ''
        return super().format(record)

# Create a stream handler with our custom formatter.
handler = logging.StreamHandler()
handler.setLevel(getattr(logging, LOGLEVEL))
formatter = CustomFormatter('[%(levelname)s] %(training)s %(message)s')
handler.setFormatter(formatter)

# Create a logger for your application.
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOGLEVEL))
# Remove default handlers if any, and add our own.
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False  # Prevent messages from being handled by the root logger.

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = self.extra.copy()
        extra.update(kwargs.get('extra', {}))
        kwargs['extra'] = extra
        return msg, kwargs

# Create an adapter with a custom "training" attribute.
training_logger = CustomAdapter(logger, {'training': '[TRAINING]'})

# Example usage:
training_logger.info("This is an informational message.")
logger.info("This is a normal info message.")
