import os


def safe_mkdir(dir_list):
    """ Creates a path or multiple paths if they do not yet exist.

    Parameters
    ----------
    dir_list: string or iterable of strings
        Contains path names to be created.

    Examples
    --------
    >>> safe_mkdir('User/Desktop/folder')
    >>> safe_mkdir(['tmp','tmp/figures','tmp2'])

    Author: Lyndon Duong 2020
    """

    def _create_path(new_dir):
        if not os.path.exists(new_dir):
            print(f"Creating path '{new_dir}'\n")
            os.makedirs(new_dir)
        else:
            print(f"Path '{new_dir}' exists! Not overwriting.\n")

    if isinstance(dir_list, (list, tuple)):
        for new_dir in dir_list:
            _create_path(new_dir)
    else:  # if input was a single string
        new_dir = dir_list
        _create_path(new_dir)
