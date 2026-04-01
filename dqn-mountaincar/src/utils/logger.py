# src/utils/logger.py

import csv
import os


class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "episode",
            "timestep",
            "return",
            "length",
            "epsilon",
            "loss",
            "mean_q",
            "success",
            "avg_return_100",   # NEW
            "buffer_size"       # NEW
        ])

    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


