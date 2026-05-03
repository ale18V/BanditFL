"""Result-file helpers."""

import os


def store_result(fd, *entries):
    fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
    fd.flush()


def make_result_file(fd, *fields):
    fd.write("# " + ("\t").join(str(field) for field in fields))
    fd.flush()
