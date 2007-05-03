def save_array(array, fname, fmt='%.18e', delimiter=' ', header=None):
    """Save array to ascii file.

       'array' is the array to be saved.

       'fname' is the filename. If the filename ends in .gz,
       the file is automatically saved in compressed gzip format.

       'fmt' is the format string to convert the array values to strings.

       'delimiter' is used to separate the fields.

       'header' if not None, is a string to be put in the top of the file.
    """
    # Open file using gzip if necessary
    if fname.endswith('.gz'):
        import gzip
        fhandle = gzip.open(fname,'wb')
    else:
        fhandle = file(fname,'w')

    # Attach header
    if header is not None:
        print >>fhandle, header + '\n'
        
    # Print array to file
    for row in array:
        print >>fhandle, delimiter.join([fmt % col for col in row]) + '\n'

def load_array(fname, comments='#', delimiter=None, converters={},
         skiprows=[], usecols=None):
    """Load array from ascii file.

       'fname' is the filename. Support for gzipped files is automatic,
       if the filename ends in .gz.

       'comments' - the character used to indicate the start of a comment
       in the file.

       'delimiter' is the character used to separate values in the
       file. None (default) implies any number of whitespaces.

       'converters' is a dictionary mapping column number to
       a function that will convert that column to a float.
       Eg, if column 0 is a chemical symbol, use:
       from ASE.ChemicalElements import numbers
       converters={0:numbers.get}.

       'skiprows' is a sequence of integer row indices to skip,
       where 0 is the first row.

       'usecols', if not None, is a sequence of integer column indices to
       use, where 0 is the first column.
    """
    # Open file using gzip if necessary
    if fname.endswith('.gz'):
        import gzip
        fhandle = gzip.open(fname)
    else:
        fhandle = file(fname)
        
    array = []
    for i, row in enumerate(fhandle):
        # Skip the desired rows
        if i in skiprows:
            continue

        # Strip comments and leading and trailing whitespaces
        row = row[:row.find(comments)].strip()

        # Skip empty rows and rows containing only comments)
        if len(row) == 0:
            continue

        cols = []
        for i, col in enumerate(row.split(delimiter)):
            # Skip the desired columns
            if i in skipcols:
                continue
            
            # Apply converters
            cols.append(converters.get(i, float)(col))            
        array.append(row)

    return array
