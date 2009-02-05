import numpy as npy

def get_handle(file, mode='r'):
    """Return filehandle correspoding to 'file'.

    'file' can be a filehandle, or a filename (string).
    Support for gzipped files is automatic, if the filename ends in .gz.
    """
    if hasattr(file, 'read'):
        fhandle = file
    else:
        if not isinstance(file, str):
            raise RuntimeError('File must be either a filehandle or a string!')
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

# We should use numpy for this! XXX
def save_array(array, file, delimiter=' ', converters={}, header=None):
    raise DeprecationWarning, 'You should use numpy.savetxt instead'
    """Save array to ascii file.

    ============== =========================================================
    Argument       Description
    ============== =========================================================
    ``array``      The array to be stored. Can be any iterable object with
                   iterable elements.
                   
    ``file``       Filehandle, or filename (string).
                   Support for gzipped files is automatic, if the filename
                   ends in .gz.

    ``delimiter``  The character used to separate fields.
    
    ``converters`` A dictionary mapping column number to a string formatter.
                   The default converter is '%.18e', but can be changed
                   by setting converters['default'] appropriately.
    
    ``header``     If not None, a string to be put in the top of the file.
    ============== =========================================================
    """
    # Open file using gzip if necessary
    fhandle = get_handle(file, 'w')

    # Determine default converter
    default_convert = converters.get('default', '%.18e')

    # Attach header
    if header is not None:
        print >>fhandle, header
        
    # Print array to file
    for row in array:
        print >>fhandle, delimiter.join(
            [converters.get(i, default_convert) % col
             for i, col in enumerate(row)])

def load_array(file, comments='#', delimiter=None, converters={},
               skiprows=[], skipcols=[], dtype='O', transpose=False):
    raise DeprecationWarning, 'You should use numpy.loadtxt instead'
    """Load array from ascii file.

    ============== ===========================================================
    Argument       Description
    ============== ===========================================================
    ``file``       Filehandle, or filename (string).
                   Support for gzipped files is automatic, if the filename
                   ends in .gz.

    ``comments``   The character used to indicate the start of a comment.

    ``delimiter``  The character used to separate values.
                   None (default) implies any number of whitespaces.
    
    ``converters`` A dictionary mapping column number to
                   a function that will convert that column string to the
                   desired type of the output array (e.g. a float).
                   Eg., if column 0 is a chemical symbol, use::

                    >>> from ASE.ChemicalElements import numbers
                    >>> converters={0:numbers.get}

                   to convert the symbol names to integer values.
                   The default converter is ``float``, but can be changed
                   by setting converters['default'] appropriately.
    
    ``skiprows``   A sequence of integer row indices to skip, where 0 is
                   the first row. Negative indices are allowed.
    
    ``skipcols``   A sequence of integer column indices to skip, where 0 is
                   the first column.
    
    ``dtype``      The dtype of the output array. Use 'list' if you do
                   not want the data array to be converted to a Numeric array.
                   The dtype 'O' (for object), should be used if not all
                   of the elements are numbers.
    
    ``transpose``  If True, will transpose output matrix, so columns can be
                   assigned to different variables. Eg.::

                    >>> col1, col2 = load('data.txt', transpose=True)
    ============== ===========================================================

    """
    # Open file using gzip if necessary
    fhandle = get_handle(file, 'r')

    # Determine default converter
    default_convert = converters.get('default', float)

   # Convert negative indices in skiprows
    skiprows = npy.array(skiprows)
    if npy.sometrue(skiprows < 0):
        lines = count_lines(fhandle)
        for i, val in enumerate(skiprows):
            if val < 0:
                skiprows[i] += lines

    array = []
    ncols = None # The number of columns in each row
    square = True # Is the data matrix square?
    for i, row in enumerate(fhandle):
        if i in skiprows:
            continue

        # Strip comments and leading and trailing whitespaces
        row = row[:row.find(comments)].strip()

        # Skip empty rows and rows containing only comments)
        if len(row) == 0:
            continue

        cols = []
        for i, col in enumerate(row.split(delimiter)):
            if i in skipcols:
                continue
            
            # Apply converters and append column
            cols.append(converters.get(i, default_convert)(col))

        # Test if data matrix is square
        if not ncols: ncols = len(cols)
        elif len(cols) != ncols: square = False

        array.append(cols)

    if not square or dtype == 'list':
        print 'Data matrix not square or dtype == list.'
        print 'Output not converted to Numeric array.'
        return array

    # Convert to Numeric array
    array = npy.array(array, dtype=dtype)

    # If single column, correct shape of array
    shape = list(array.shape)
    try:
        shape.remove(1)
    except ValueError:
        pass
    else:
        array.shape = tuple(shape)
    
    if transpose:
        array = npy.transpose(array)
    
    return array

if __name__ == '__main__':
    def print_file(name):
        for line in open(name):
            print line[:-1]
        
    square_array = [['hallo', 1],
                    ['world', 2]]

    non_square = [[5,4,3],
                  [4,3],
                  [6,4,1, 'hey']]

    # test save of square array
    save_array(square_array, 'test.dat', header='#String number',
               converters={'default': '%7s', 1: '%6.1f'})
    print_file('test.dat')

    # test load of square array
    names, values = load_array('test.dat', converters={'default': str},
                               transpose=True)
    print zip(names, values)

    # test save of non-square array
    save_array(non_square, 'test.dat', converters={3: '%s'})
    print_file('test.dat')

    # test load of non-square array
    print load_array('test.dat', skipcols=[0], skiprows=[-2],
                     converters={3:str})
    
