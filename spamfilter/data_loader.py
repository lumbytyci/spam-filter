def load_data_from_file(path: str) -> tuple:
    """Load data from spam collection text file

    Parameters:
    path (str): Path to text file

    Returns:
    tuple: Returns tuple with labels and emails
    """

    with open(path, 'r') as spam_collection_file:
        labels = []
        emails = []
        for line in spam_collection_file:
            label, email = line.strip().split(maxsplit=1)
            labels.append(label.strip())
            emails.append(email.strip())

    return labels, emails
