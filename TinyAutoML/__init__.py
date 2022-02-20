import os
from pathlib import Path

if __name__=='__main__':
    if not "root" in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
    os.chdir(root)