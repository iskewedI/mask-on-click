import threading
import traceback
import logging

logger = logging.getLogger(__name__)


class ThreadOperation:
    def __init__(self, callback):
        self.result = None
        self.callback = callback
        self.error = False

    def start(self):
        thread = threading.Thread(target=self.inpaint)
        try:
            thread.start()
        except Exception as e:
            self.handle_error(e)

    def inpaint(self):
        try:
            self.result = self.callback()
        except Exception as e:
            self.handle_error(e)

    def handle_error(self, error):
        traceback.print_exc()
        logger.exception("Error on ThreadOperation => ", error)
        self.error = True
