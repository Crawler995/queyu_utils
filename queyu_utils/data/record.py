import csv
import os
import shutil
import json
from typing import List, Any

from ..common.others import get_cur_time_str


class CSVDataRecord:
    """Util for save critical data as CSV format safely.

    If we use the normal way to save data as CSV format, we gonna write code like this:
        >>> with open(csv_file_path, 'w') as f:
        >>>     f.write(...)
    Or use built-in util ``csv.writer(f)`` to do this. Anyway, we can't prevent from open the csv file via 'w',
    which will flush all content in it if it already exists. Sometimes we will lose some valuable data because of this.
    This class is dedicated to solve this problem.

    Examples:
        >>> # Run this code firstly to run experiments and record some important data.
        >>> csv_data_record = CSVDataRecord('./1.csv', ['model_name', 'acc'], backup=True)
        >>> csv_data_record.write(['vgg16', 0.877])
        >>> csv_data_record.write(['resnet18', 0.912])
        >>> # If run this code again, the previous CSV file will be backup to prevent from losing valuable data.

    Args:
        file_path: Target CSV file path.
        header: CSV header.
        backup: Whether to backup the existed CSV file. If backup, the existed file will be backup to
            ``{file_path}.{cur_time}.backup``.
    """
    def __init__(self, file_path: str, header: List[str], backup: bool = True):
        self.file_path = file_path
        self.header = header

        if backup and os.path.exists(file_path):
            backup_file_path = '{}.{}.backup'.format(file_path, get_cur_time_str())
            shutil.copyfile(file_path, backup_file_path)

        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def write(self, data: List[Any]):
        """Write a row of data to CSV file.

        Args:
            data: A row of data.
        """
        assert len(data) == len(self.header)

        with open(self.file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)


class JSONDataRecord:
    """Util for save critical data as JSON format safely. (Has the same function with :class:`CSVDataRecord`)

    If the critical data is saved as dict (object) in the code,
    and it is changed iteratively during your experiment,
    it's necessary to save the dict in each iteration to ensure the safety of data.

    Args:
        file_path: Target JSON file path.
        backup: Whether to backup the existed JSON file. If backup, the existed file will be backup to
            ``{file_path}.{cur_time}.backup``.
    """
    def __init__(self, file_path: str, backup: bool = True):
        self.file_path = file_path
        self.backup = backup

        if self.backup and os.path.exists(self.file_path):
            backup_file_path = '{}.{}.backup'.format(self.file_path, get_cur_time_str())
            shutil.copyfile(self.file_path, backup_file_path)

    def write(self, obj: Any, readable: bool = True):
        """Write the object into the JSON file.

        Args:
            obj: Target object.
            readable: If readable, the reasonable indent will be added into the JSON content.
        """
        with open(self.file_path, 'w') as f:
            obj_str = json.dumps(obj, indent=2 if readable else None)
            f.write(obj_str)
