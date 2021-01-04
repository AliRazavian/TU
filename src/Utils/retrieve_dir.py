import os
from typing import Optional
from difflib import get_close_matches


def strip_trailing_slash(path: str):
    if len(path) == 0:
        return path

    if path[:-1] == '/':
        return path[:-1]

    return path


def strip_initial_slash(path: str):
    if len(path) == 0:
        return path

    if path[0] == '/':
        return path[1:]

    return path


def find_closest_matching_dirs(path: str, base_path: Optional[str] = None):
    path = strip_trailing_slash(path)

    needle = os.path.basename(path)
    if needle == '':
        return (None, None)

    parent_path = path[:-len(needle)]
    if parent_path == '':
        if base_path is None or not os.path.exists(base_path):
            return (None, None)

        return (get_close_matches(needle, sorted([d for d in os.listdir(base_path)])), base_path)

    search_path = parent_path
    if base_path is not None:
        search_path = os.path.join(base_path, parent_path)

    if os.path.isdir(search_path):
        return (get_close_matches(needle, sorted([d for d in os.listdir(search_path)])), parent_path)

    return find_closest_matching_dirs(parent_path)


def retrieve_dir(path: str, base_path: Optional[str] = None, expected_depth: int = 0):
    """
    Retrieves directory with autocomplete if the directory is incomplete.
    If the directory name is ambigous it raises an FileNotFoundError.

    path (str): The path that we want to find e.g. 20200212/14.52
    base_path (str): The base path for that dir if any.
    expected_depth (int): If we want the returned directory to be a subdir, e.g. we only have one run that
                          date and we don't want to bother with writing subdirectories then it should be fine
                          with just the date and if there is one subdir we enter it in a recursive manner
                          until expected_depth == 0
    """
    full_path = strip_trailing_slash(path)

    if base_path is not None:
        full_path = os.path.join(base_path, full_path)

    if os.path.exists(full_path):
        ret_path = full_path
        if base_path is not None:
            ret_path = full_path[len(base_path):]

        if expected_depth is not None or expected_depth <= ret_path.count('/'):
            return strip_initial_slash(ret_path)

        subdirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
        if len(subdirs) != 1:
            raise FileNotFoundError(f'Could find the directory {full_path} but there should be a single subdir' +
                                    f' unfortunately found {len(subdirs)} subdirectories.')

        return retrieve_dir(path=os.path.join(full_path, subdirs[0]),
                            base_path=base_path,
                            expected_depth=expected_depth - 1)

    parent_path = os.path.dirname(full_path)
    if parent_path == '':
        raise Exception(f'Could not identify a matching directory to {path} (subfolder of {base_path})')

    if os.path.exists(parent_path):
        needle = os.path.basename(full_path)
        available_dirs = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
        subdirs = [d for d in available_dirs if d[:len(needle)] == needle]
        if len(subdirs) != 1:
            close_matches = get_close_matches(needle, available_dirs)
            raise FileNotFoundError(f'Could find the directory {full_path} with subdir needle {needle}' +
                                    f' the closes matches are {", ".join(close_matches)}.')

        ret_path = full_path
        if base_path is not None:
            ret_path = ret_path[len(base_path):]
        return strip_initial_slash(os.path.join(ret_path, subdirs[0]))

    # No matching found - we need to raise an error
    matches, dirname = find_closest_matching_dirs(path=path, base_path=base_path)
    if matches is None:
        raise FileNotFoundError(f'Could not find the file path {path} (subfolder of {base_path})')

    raise FileNotFoundError(f'The colsest matching path for {path} seems to be in {dirname}: ' + ', '.join(matches))
