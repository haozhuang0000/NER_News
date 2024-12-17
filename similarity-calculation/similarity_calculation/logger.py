import logging
import time
import os


class Log(object):
    def __init__(self, logger=None):
        # create a logger
        self.name = logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        # create a handler
        self.log_time = time.strftime("%Y%m%d")
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../log")

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.file_dir = file_dir
        self.log_path = os.path.join(
            self.file_dir, self.name + "_" + self.log_time + ".log"
        )

        fh = logging.FileHandler(self.log_path, "a", encoding="utf-8")
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        fh.close()
        ch.close()

    def getlog(self):
        return self.logger

    def getpath(self):
        return self.log_path
