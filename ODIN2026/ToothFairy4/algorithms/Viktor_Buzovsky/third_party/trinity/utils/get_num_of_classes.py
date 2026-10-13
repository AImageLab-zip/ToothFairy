from .labels_remap import LABEL_MAPPING

def get_num_of_classes() -> int:
    """
    Returns the number of classes after remapping.
    0 is background, 1-77 are foreground classes.
    Total: 78
    """
    return len(set(LABEL_MAPPING.values()))

if __name__ == "__main__":
    print(f"Number of classes (remapped): {get_num_of_classes()}")
