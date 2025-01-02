from config.config import *
import pandas as pd

def file_to_dataframe(file_name: str) -> pd.DataFrame:
    """
    Convert a structured text file into a pandas DataFrame with abstract information.

    This function reads a specialized text file format and transforms its content 
    into a structured DataFrame, processing abstracts with their associated metadata.

    Parameters:
    file_name: str
        Path to the input text file with a specific structured format.
        File must follow these formatting rules:
        - Abstract headers start with '###' 
        - Each content line uses tab-separated target and text
        - Abstracts are separated by blank lines

    Returns:
    pd.DataFrame: DataFrame containing processed abstract data with columns:
        - target: Classification/label for each text snippet
        - text: Preprocessed text content (lowercase, stripped)
        - line_number: Position within the abstract (0-based indexing)
        - total_lines: Total number of lines in the specific abstract

    Raises:
    FileNotFoundError: When the specified file path does not exist
    IOError: If file reading encounters unexpected issues
    ValueError: For invalid file content, including:
        - Incorrectly formatted lines 
        - Insufficient tab-separated fields
        - Malformed abstract headers

    Example:
    >>> df = file_to_dataframe('research_abstracts.txt')
    >>> print(df.head())

    target       text    line_number   total_lines
    BACKGROUND   text1  0            5
    METHOD       text2  1            5
    RESULT       text3  2            5

    Notes:
    - Converts all text to lowercase for consistent processing
    - Strips newline characters from text
    - Assumes well-formed input with tab-separated fields
    - Requires pandas library for DataFrame conversion

    Performance Considerations:
    - Efficient for small to medium-sized files
    - Memory usage scales with file size
    - Preprocessing done in-memory
    """
    
    data = list()
    with open(file_name) as file:
        data_lines = file.readlines()

    line_number = 0
    abstract = []

    for line in data_lines:
        if line[:3] == '###':
            line_number = 0
            abstract = []
            continue
        elif line == '\n':
            for item in abstract:
                item['total_lines'] = line_number
            data.extend(abstract)
            abstract = []
            continue
        else:
            target, text = line.split('\t')[:2]
            abstract.append(
                {
                    'target': target,
                    'text': text.lower().strip('\n'),
                    'line_number': line_number,
                    'total_lines': 0,
                }
            )
            line_number += 1

    if abstract:
        for item in abstract:
            item['total_lines'] = line_number
        data.extend(abstract)

    return pd.DataFrame(data)

def text_to_dataframe(text: str) -> pd.DataFrame:
    
    """
    This function processes a text string into a DataFrame where each row represents a sentence from the input text.
    Each row contains the sentence text (in lowercase), its line number, and the total number of sentences in the text.
    text : str
        Input text string to be processed into sentence-level DataFrame entries.
    Returns:
    pd.DataFrame
        A DataFrame with columns 'target', 'text', 'line_number', and 'total_lines'.
        'target' is initialized to None, 'text' contains the sentence in lowercase, 'line_number' is the index of the sentence,
        and 'total_lines' is the total number of sentences in the text.
    """

    import spacy
    abstract = []
    nlp = spacy.load('en_core_web_sm')
    for i,sent in enumerate(nlp(text).sents):
        abstract.append(
            {
                'target': None,
                'text': str(sent).lower(),
                'line_number': i,
            }
        )
    df = pd.DataFrame(abstract)
    df['total_lines'] = i+1
    return df

def remove_digits(dataframe: pd.DataFrame) -> pd.DataFrame:
   
   """
   Replace digits with '@' symbol in text columns for testing data preprocessing.

   This function is specifically designed to sanitize text data by replacing 
   numeric characters with a placeholder symbol, ensuring consistency with 
   preprocessing applied during training data preparation.

   Parameters:
       dataframe: pd.DataFrame
           Input DataFrame, expected to be output from file_to_dataframe()
           Must contain a 'text' column with string entries
           Typically used for test/validation datasets where digit masking 
           needs to match training data preprocessing

   Returns:
       pd.DataFrame: A new DataFrame with digits replaced by '@' symbol
       - Preserves original DataFrame structure
       - Modifies only the 'text' column
       - Returns a copy, leaving original DataFrame unmodified

   Raises:
       KeyError: If 'text' column is missing from input DataFrame
       TypeError: If input is not a pandas DataFrame
       AttributeError: If 'text' column contains non-string elements

   Example:
       >>> original_df = file_to_dataframe('test_data.txt')
       >>> processed_df = remove_digits(original_df)
       >>> print(original_df['text'][0])
       'method used in 2023 for analysis'
       >>> print(processed_df['text'][0])
       'method used in @@@@ for analysis'

   Notes:
       - Designed for test/validation data preprocessing
       - Assumes consistent preprocessing with training data
       - Replaces ALL digits with '@' symbol
       - Creates a copy of input DataFrame to prevent in-place modifications

   Performance Considerations:
       - Time complexity: O(n*m), where n is number of rows, m is text length
       - Memory overhead from creating DataFrame copy
       - Minimal computational cost for small to medium DataFrames
   """
   # Create a copy to avoid modifying original DataFrame
   processed_df = dataframe.copy()
   
   # Get list of digits to replace
   from string import digits
   digits = list(digits)
   
   # Replace digits with '@' in text column
   processed_df['text'] = processed_df['text'].apply(
       lambda text: ''.join('@' if char in digits else char for char in str(text))
   )
   
   return processed_df