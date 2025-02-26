import pandas as pd
import pathlib
import os

def read_file(file_name : str )  -> pd.DataFrame:
        """
        summary
        """
        try:
            dir_folder = pathlib.Path().cwd().parent
            print(dir_folder)
            file_path  = dir_folder / "data" 
            print(file_path)
            df = pd.read_csv(os.path.join(file_path/file_name))
            return df
        except FileNotFoundError:
            print(f"Error: The file at '{file_name}' was not found.")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")