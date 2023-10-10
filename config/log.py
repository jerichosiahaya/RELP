import logging

class LogFormatter(logging.Formatter):
    def format(self, record):
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname
        logger = record.name
        message = record.getMessage().replace('"', '\\"')
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "logger": logger,
            "message": message
        }
        return f'{{"timestamp":"{timestamp}", "level":"{level}", "logger":"{logger}", "message":"{message}"}}'

def log(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = LogFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
