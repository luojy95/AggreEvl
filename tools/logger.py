from tools.writer import writer

from datetime import datetime


class Logger:
    LOGGER_PREFIX = ""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def __init__(self, filepath=None) -> None:
        if filepath:
            writer.check_out_dir(filepath)
            self.f = open(filepath, "w+", encoding="utf-8")
        else:
            self.f = None

    def get_time(self):
        now = datetime.now()
        return now.strftime("%H:%M:%S")

    def info(self, message: str):
        if self.f:
            self.f.writelines([message + "\n"])
        print(f"({self.get_time()})" + self.LOGGER_PREFIX + "[Info] - " + message)

    def warning(self, message: str):
        if self.f:
            self.f.writelines([message + "\n"])
        print(
            self.WARNING
            + f"({self.get_time()})"
            + self.LOGGER_PREFIX
            + "[Warning] - "
            + message
            + self.ENDC
        )

    def error(self, message: str):
        if self.f:
            self.f.writelines([message + "\n"])
        print(
            self.FAIL
            + self.BOLD
            + f"({self.get_time()})"
            + self.LOGGER_PREFIX
            + "[Fatal] - "
            + message
            + self.ENDC
        )

    def ok(self, message: str):
        if self.f:
            self.f.writelines([message + "\n"])
        print(
            self.OKGREEN
            + f"({self.get_time()})"
            + self.LOGGER_PREFIX
            + "[Success] - "
            + message
            + self.ENDC
        )


default_logger = Logger()


def test():
    logger = Logger()
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.ok("This is a success message")


if __name__ == "__main__":
    test()
