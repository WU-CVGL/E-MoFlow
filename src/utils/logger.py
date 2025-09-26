import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from colorama import Fore, Style, Back, init

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    '''
        color mapping for different log levels
    '''
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.LIGHTRED_EX + Style.BRIGHT,
    }
    TIME_COLOR = Fore.LIGHTBLACK_EX  
    FILE_COLOR = Fore.LIGHTMAGENTA_EX   
    MESSAGE_COLOR = Fore.WHITE       
    
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
    
    def format(self, record):
        record_copy = logging.makeLogRecord(record.__dict__)
        
        level_color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        asctime = self.formatTime(record_copy, self.datefmt)
        levelname = record_copy.levelname
        filename = record_copy.filename
        lineno = record_copy.lineno
        message = record_copy.getMessage()
        colored_time = f"{self.TIME_COLOR}{asctime}{Style.RESET_ALL}"
        colored_level = f"{level_color}{levelname:<8}{Style.RESET_ALL}"  
        colored_file = f"{self.FILE_COLOR}{filename}:{lineno}{Style.RESET_ALL}"
        colored_message = f"{level_color}{message}{Style.RESET_ALL}"
        
        return f"{colored_time} - {colored_level} - {colored_file} - {colored_message}"

def setup_logger(
    name="debug_logger",
    logs_dir="logs",
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
    max_file_size=10*1024*1024,  # 10MB
    backup_count=5,
):
    """
    Initialize a logger with both console and file handlers.
    
    Args:
        name: name of the logger
        logs_dir: directory for saved log files
        console_level: console output level
        file_level: file output level
        max_file_size: maximum size of a single log file (bytes)
        backup_count: number of backup files
        use_advanced_colors: whether to use advanced color formatting
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m%d_%H:%M")
    log_file = logs_dir / f"{timestamp}.log"
    
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # File format (wo colors)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console format (with colors)
    console_formatter = ColoredFormatter(
        datefmt='%H:%M:%S'
    )

    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger has been initialized and log is saved as: {log_file}")
    
    return logger

_global_logger = None

def get_logger():
    """get global logger instance, if not exists, create one"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Style.RESET_ALL}"

def red(text: str) -> str:
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def green(text: str) -> str:
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def yellow(text: str) -> str:
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

def lred(text: str) -> str:
    return f"{Fore.LIGHTRED_EX}{text}{Style.RESET_ALL}"