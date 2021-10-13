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
