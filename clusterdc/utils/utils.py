import pandas as pd
import importlib.resources as pkg_resources


def read_file(file_path):
    """
    Reads a file (.xlsx, .xls, or .csv) into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the input file.
        
    Returns:
        DataFrame: The data read from the file.
    """
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .xlsx, .xls, or .csv file.")

def write_file(data, file_path):
    """
    Writes a pandas DataFrame to a file (.xlsx or .csv).
    
    Parameters:
        data (DataFrame): The data to write.
        file_path (str): Path to the output file.
    """
    if file_path.endswith(".xlsx"):
        data.to_excel(file_path, index=False, engine='openpyxl')
    elif file_path.endswith(".csv"):
        data.to_csv(file_path, index=False)
    else:
        raise ValueError("Unsupported file format. Please specify a .xlsx or .csv file.")

# Example usage
if __name__ == "__main__":
    # Reading a file
    df = read_file("example.xlsx")
    print(df)
    
    # Writing to a file
    write_file(df, "output.csv")

def load_training_data():
    """
    Load training data from the package's data directory.

    Returns:
        DataFrame: The training data as a pandas DataFrame.
    """
    with pkg_resources.open_text("clusterdc.data", "training_data.csv") as f:
        return pd.read_csv(f)