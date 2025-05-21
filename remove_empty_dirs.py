import os

def remove_empty_dirs(root_dir):
    # Walk the directory tree in bottom-up order
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # If the directory is empty (no files and no dirs)
        if not dirnames and not filenames:
            print(f"Removing empty directory: {dirpath}")
            os.rmdir(dirpath)

if __name__ == "__main__":
    root = "."  # Change this to your root directory
    remove_empty_dirs(root)