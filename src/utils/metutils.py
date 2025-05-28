import pathlib

def get_files_pathlib(folder_path, extension=None):
    """
    Gets a list of files in a folder using pathlib, optionally filtering by file extension.
    Args:
        folder_path (str or Path): Path to the folder.
        extension (str, optional): File extension to filter by (e.g., 'geojson' or 'tif'). Case-insensitive.
    Returns:
        List[pathlib.Path]: List of Path objects for the matched files.
    """
    folder = pathlib.Path(folder_path)

    if not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory.")
        return []

    files = [
        entry for entry in folder.iterdir()
        if entry.is_file() and (extension is None or entry.suffix.lower() == f'.{extension.lower()}')
    ]
    return files
