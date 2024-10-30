import ijson
import json
import random
from pathlib import Path
from typing import List, Any, Dict


def load_json_files(dir: str) -> List[Any]:
    """Find all files containing series of JSON objects and load their contents.

    Uses ijson for memory-efficient streaming parsing.

    Args:
        directory (str): Path to the directory containing JSON files

    Returns:
        List[Any]: Combined list of data from all JSON files

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    dir_path = Path(dir)
    if not dir_path.exists():
        raise FileNotFoundError(f'Directory not found: {dir}')

    combined_data = []
    # Process each JSON file in the directory
    for json_file in dir_path.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                combined_data.extend(ijson.items(f, '', multiple_values=True))
        except Exception as e:
            print(f'Error processing {json_file}: {str(e)}')

    if not combined_data:
        raise(f'No data found under directory: {dir}')

    return combined_data


def partition_k(objects: List[Any], k: int) -> Dict[int, List[Any]]:
    """Partition a list into k approximately equal-sized groups.

    The function distributes any remainder elements across the first few partitions
    to ensure the most balanced distribution possible.

    Args:
        objects: List of elements to partition
        k: Number of partitions to create (must be positive)

    Returns:
        Dictionary mapping partition indices (0 to k-1) to lists of elements

    Raises:
        ValueError: If k <= 0 or k > len(objects)
    """
    if k <= 0:
        raise ValueError("Number of partitions must be positive")
    if k > len(objects):
        raise ValueError("Cannot create more partitions than elements")

    base_size = len(objects) // k
    remainder = len(objects) % k
    return {i: objects[i * base_size + min(i, remainder):
                       (i + 1) * base_size + min(i + 1, remainder)]
            for i in range(k)}


def write_json_objects(objects: List[Any], filename: str):
    """Write a list of JSON objects to a file, one object per line.

    Args:
        objects: List of dictionaries/objects to write.
        filename: Output file name
    """
    output_path = Path(filename)
    # Create Parent dirs in filename if they don't exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        for obj in objects:
            json_str = json.dumps(obj)
            f.write(json_str + '\n')
