import logging


class LogFilter(logging.Filter):
    FILTERED_WORDS = ["http://127.0.0.1:7861/"]

    def filter(self, record):
        recordMsg = record.getMessage()

        for word in LogFilter.FILTERED_WORDS:
            if word in recordMsg:
                return False

        return True


def config_logs():
    logging.basicConfig(filename="trace.log", encoding="utf-8", level=logging.INFO)
    filter = LogFilter()

    # Add filter to all handlers
    for handler in logging.root.handlers:
        handler.addFilter(filter)
