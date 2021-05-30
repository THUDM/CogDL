import unicodedata
from tqdm import tqdm

COLORCODES = {
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "reset": "\x1b[0m",
}


def colored(text, color):
    return COLORCODES.get(color, "") + text + COLORCODES.get("reset", "")


OAG_TOKEN_TYPE_NAMES = ["TEXT", "AUTHOR", "VENUE", "AFF", "FOS", "FUND"]


def stringLenCJK(string):
    return sum(1 + (unicodedata.east_asian_width(c) in "WF") for c in string)


def stringRjustCJK(string, length):
    return " " * (length - stringLenCJK(string)) + string


class MultiProcessTqdm(object):
    def __init__(self, lock, positions, max_pos=100, update_interval=1000000, leave=False, fixed_pos=False, pos=None):
        self.lock = lock
        self.positions = positions
        self.max_pos = max_pos
        self.update_interval = update_interval
        self.leave = leave
        self.pbar = None
        self.pos = pos
        self.fixed_pos = fixed_pos

    def open(self, name, **kwargs):
        with self.lock:
            if self.pos is None or not self.fixed_pos:
                self.pos = 0
                while self.pos in self.positions:
                    self.pos += 1
                self.positions[self.pos] = name
            self.pbar = tqdm(
                position=self.pos % self.max_pos, leave=self.leave, desc="[%2d] %s" % (self.pos, name), **kwargs
            )
        self.cnt = 0

    def reset(self, total, name=None, **kwargs):
        if self.pbar:
            with self.lock:
                if name:
                    self.pbar.set_description("[%2d] %s" % (self.pos, name))
                self.pbar.reset(total=total)
                self.cnt = 0
        else:
            self.open(name=name, total=total, **kwargs)

    def set_description(self, name):
        with self.lock:
            self.pbar.set_description("[%2d] %s" % (self.pos, name))

    def update(self, inc: int = 1):
        self.cnt += inc
        if self.cnt >= self.update_interval:
            with self.lock:
                self.pbar.update(self.cnt)
            self.cnt = 0

    def close(self):
        with self.lock:
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            if self.pos in self.positions:
                del self.positions[self.pos]
