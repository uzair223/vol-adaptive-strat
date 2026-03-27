import logging
from contextlib import contextmanager
import sys
from typing import Optional

import logging
from contextlib import contextmanager
from typing import Optional

@contextmanager
def suppress(level=logging.WARNING, module: Optional[str] = None):
    if module is None:
        # Global suppression
        previous = logging.root.manager.disable
        logging.disable(level)
        try:
            yield
        finally:
            logging.disable(previous)
    else:
        # Module-specific suppression
        logger = logging.getLogger(module)
        old_level = logger.level
        logger.setLevel(level + 1)  # suppress <= level
        try:
            yield
        finally:
            logger.setLevel(old_level)
            
def setup_logging() -> None:
    """Configure logging for dry run."""
    formatter = logging.Formatter(
        "\x1b[90m%(asctime)s \x1b[35m[%(name)s %(levelname)s] \x1b[00m%(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)