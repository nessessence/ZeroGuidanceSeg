from __future__ import absolute_import
import os
import sys
import errno


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class Logger(object):
    def __init__(self, fpath=None, append=False,overwrite=True):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            if os.path.isfile(fpath) and append:
                self.file = open(fpath, 'a')
            elif not overwrite: 
                mkdir_if_missing(os.path.dirname(fpath))
                self.file = open(fpath, 'w')
            else: 
                os.makedirs(os.path.dirname(fpath),exist_ok=True)
                self.file = open(fpath, 'w')
    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
