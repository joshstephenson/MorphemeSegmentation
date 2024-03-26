#!/usr/bin/env python

import sys
import re


def postprocess_chars(line):
    # remove whitespace
    line = "".join(line.strip().split(" "))
    # replace sentencepiece marker (and normal underscore) with a space
    line = line.replace('▁', ' ')
    line = line.replace("_", " ")

    # change morpheme boundary markers (including preceding whitespace)
    line = re.sub(r"\s*\|", " @@", line)
    line = line.strip()
    return line


def main():
    for line in sys.stdin:
        sys.stdout.write(postprocess_chars(line) + "\n")


if __name__ == "__main__":
    main()
