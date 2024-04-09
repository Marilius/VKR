from os import listdir, remove
from os.path import isfile, join


data_dirs = [
    r'./results',
]


def delete_recursive(path: str) -> None:
    try:
        if isfile(path):
            remove(path)
        else:
            for file in listdir(path):
                file_path = join(path, file)
                delete_recursive(file_path)
            remove(path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


if __name__ == '__main__':
    for dir in data_dirs:
        delete_recursive(dir)
