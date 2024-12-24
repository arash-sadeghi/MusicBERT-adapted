import os

def find_midi_files(directory_path):
    """
    Recursively find all .midi files in a given directory and its subdirectories.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list: A list containing the absolute paths of all .midi files.
    """
    midi_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.midi') or file.lower().endswith('.mid'):
                midi_files.append(os.path.abspath(os.path.join(root, file)))

    return midi_files

# directory = "/Users/arashsadeghiamjadi/Desktop/WORKDIR/Drum_transformer/dataset/groove"
# midi_files = find_midi_files(directory)
# print(midi_files)
