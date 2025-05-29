#region Libraries

#%%
import pathlib

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def get_files_pathlib(folder_path: str, extension: str = None) -> list:
    '''Get a list of files in a folder path.

    Args:
        folder_path (str): Folder path.
        extension (str, optional): Extension to filter. Defaults to None.

    Returns:
        list: List of file paths.
    '''
    folder = pathlib.Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return []
    # Using a list comprehension for conciseness
    # .is_file() checks if the entry is a file (not a directory or symlink to a dir)
    if extension is None:
        files = [entry for entry in folder.iterdir() if entry.is_file()]
    else:
        files = [entry for entry in folder.iterdir() if entry.suffix == f'.{extension}']
    return files # Returns a list of Path objects

#endregion -----------------------------------------------------------------------------------------
