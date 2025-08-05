import os

def test_output_plot_exists():
    """Check if the poisoning accuracy plot has been generated."""
    assert os.path.exists("poisoning_accuracy_plot.png"), "Plot file not found!"

def test_data_folder_exists():
    """Check if the data folder exists."""
    assert os.path.isdir("data"), "Data folder not found!"