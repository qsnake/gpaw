import Numeric as num

def get_handle(file, mode='r'):
    """Return filehandle correspoding to 'file'.

       'file' can be a filehandle, or a filename (string).
       Support for gzipped files is automatic, if the filename ends in .gz.
    """
    if hasattr(file, 'read'):
        fhandle = file
    else:
        assert type(file) == str, 'file must be either filehandle or a string'
        if file.endswith('.gz'):
            import gzip
            mode += 'b'
            fhandle = gzip.open(file, mode)
        else:
            fhandle = open(file, mode)
    
    return fhandle

def count_lines(file):
    """Count the number of lines in 'file'

       'file' can be a filehandle, or a filename (string).
       Support for gzipped files is automatic, if the filename ends in .gz.
    """
    if hasattr(file, 'read'):
        fname = file.name
    else:
        assert type(file) == str, 'file must be either filehandle or a string'
        fname = file
    
    fh = get_handle(fname, 'r')
    lines = 0
    for line in fh:
        lines += 1
    return lines

def save_array(array, file, fmt='%.18e', delimiter=' ', header=None):
    """Save array to ascii file.

       'array' is the array to be saved.

       'file' can be a filehandle, or a filename (string).
       Support for gzipped files is automatic, if the filename ends in .gz.

       'fmt' is the format string to convert the array values to strings.

       'delimiter' is used to separate the fields.

       'header' if not None, is a string to be put in the top of the file.
    """
    # Open file using gzip if necessary
    fhandle = get_handle(file, 'w')

    # Attach header
    if header is not None:
        print >>fhandle, header + '\n'
        
    # Print array to file
    for row in array:
        print >>fhandle, delimiter.join([fmt % col for col in row]) + '\n'

def load_array(file, comments='#', delimiter=None, converters={},
         skiprows=[], skipcols=[]):
    """Load array from ascii file.

       'file' can be a filehandle, or a filename (string).
       Support for gzipped files is automatic, if the filename ends in .gz.

       'comments' is the character used to indicate the start of a comment
       in the file.

       'delimiter' is the character used to separate values in the
       file. None (default) implies any number of whitespaces.

       'converters' is a dictionary mapping column number to
       a function that will convert that column to a float.
       Eg, if column 0 is a chemical symbol, use:
       from ASE.ChemicalElements import numbers
       converters={0:numbers.get}.

       'skiprows' is a sequence of integer row indices to skip,
       where 0 is the first row. Negative indices are allowed.

       'skipcols', is a sequence of integer column indices to skip,
       where 0 is the first column.
    """
    # Open file using gzip if necessary
    fhandle = get_handle(file, 'r')

    # Convert negative indices in skiprows
    skiprows = num.array(skiprows)
    if num.sometrue(skiprows < 0):
        lines = count_lines(fhandle)
        for i, val in enumerate(skiprows):
            if val < 0:
                skiprows[i] += lines

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
        array.append(cols)

    # Convert to Numeric array if possible
    array = num.array(array)
    try:
        array = num.array(array)

        # If single row or single column, correct shape of array
        shape = list(array.shape)
        try:
            shape.remove(1)
            array.shape = tuple(shape)
        except ValueError:
            pass
    except TypeError:
        print 'Data matrix not square, unable to make Numeric array'
    
    return array
