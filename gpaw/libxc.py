class Libxc:
    """Class provides methods to retrieve functionals from libxc library."""
    def __init__(self):
        self.version = 3496 # svn version of libxc
        # this below is used both as dictionary name, and python module name
        self.libxc_functionals_file = 'libxc_functionals'

    def lxc_split_xcname(self, xcname):
        """Return functional symbols for libxc, based on xcname.
        The format of xcname and resulting functional symbols are:
        XC (xc, None, None) (for example XC_LB)
        X-C (None, X, C) (for example X_PBE-C_LYP).
        See libxc/src/xc_funcs.h (or libxc_functionals.py) for valid symbols -
        the part after second underline is used from the #define.
        """
        xc, x, c = None, None, None
        assert isinstance(xcname, str)
        if xcname.find('-') == -1:
            xc = xcname
            assert (len(xc) > 0)
        else:
            x, c = xcname.split('-')
            assert (len(x) > 0) and (len(c) > 0)
        return xc, x, c

    def get_lxc_functional(self, symbols, file=None):
        """Return functional identifiers, based on symbols
        obtained from lxc_split_xcname call, by:
        1) parsing #define from an c include file
        (the full path to the file must be given),
        1) using dictionary from libxc_functionals.py.
        The 1) method can be used only in the development
        version of gpaw with the access to the libxc source code.
        """
        # set scanning function
        if file is None:
            file = self.libxc_functionals_file
            scanning_function = self.get_lxc_identifier_from_py_file
        else:
            scanning_function = self.get_lxc_identifier_from_h_file
        assert len(symbols) == 3
        functionals = []
        for functional in symbols:
            if (functional == 'None'): # if specified in the input
                # allow for exchange- or correlation-only functionals
                functionals.append(-1)
            else:
                functionals.append(scanning_function(functional, file))
        return tuple(functionals)

    def get_lxc_identifier_from_py_file(self, identifier, file):
        """Extract the value of #define from the libxc_functionals.py file."""
        assert file is not None
        from gpaw.libxc_functionals import libxc_functionals # MDTMP hard fix!
        value = -1 # assume no corresponding define found
        for key in libxc_functionals.keys():
            # compare with the identifier after second underline
            define = key
            define = define[define.find('_')+1:]
            define = define[define.find('_')+1:]
            if identifier == define:
                # extract the value
                value = int(libxc_functionals[key])
                break
        if identifier is not None:
            assert (value != -1), 'XC functional not found'
        assert isinstance(value, int), 'XC functional code not integer number'
        return value

    def get_lxc_identifier_from_h_file(self, identifier, file):
        """Extract the value of #define from an c include file."""
        f = open(file, 'r')
        lines = filter(self.lxc_define_filter, f.readlines())
        assert len(lines) > 0
        value = -1 # assume no corresponding define found
        for line in lines:
            # compare with the identifier after second underline
            define = line.split(None, 2)[1]
            define = define[define.find('_')+1:]
            define = define[define.find('_')+1:]
            if identifier == define:
                # extract the value stripping c comments
                value = int(line.split(None, 2)[2].split(None, 1)[0])
                break
        if identifier is not None:
            assert (value != -1), 'XC functional not found'
        assert isinstance(value, int), 'XC functional code not integer number'
        return value

    def lxc_define_filter(self, s):
        return (
            s.startswith('#define  XC_LDA') or
            s.startswith('#define  XC_GGA')
            ## XC_MGGA and XC_LCA not implemented yet # MDTMP
            ##            s.startswith('#define  XC_MGGA') or  # MDTMP
            ##            s.startswith('#define  XC_LCA')  # MDTMP
            ## End of: XC_MGGA and XC_LCA not implemented yet  # MDTMP
            )

    def construct_libxc_functionals_dict(self, file='../c/libxc/src/xc_funcs.h'):
        """Method for generating the dictionary libxc_functionals.
        Should be used only at 'python setup.py'"""
        txt = '# Computer generated code! Hands off!\n'
        txt += '# libxc: svn version ' + str(self.version) + '\n'
        txt += '# http://www.tddft.org/programs/octopus/wiki/index.php/Libxc\n'
        from os.path import abspath
        # Find the full path to file
        file = abspath(file)
        f = open(file, 'r')
        lines = filter(self.lxc_define_filter, f.readlines())
        assert len(lines) > 0
        # Start libxc_functionals dictionary
        txt += str(self.libxc_functionals_file) + ' = {\n'
        # Put all the defines into the dictionary
        for line in lines:
            # extract the define
            define = line.split(None, 2)[1]
            # extract the value stripping c comments
            value = int(line.split(None, 2)[2].split(None, 1)[0])
            txt += "'" + str(define) + "'" + ': ' + str(value) + ',\n'
        # replace last coma+(end of line) with nice dictionary closing
        txt = txt[:-2] + '\n}'
        libxc_functionals_file = 'gpaw/'+self.libxc_functionals_file+'.py'
        f = open(libxc_functionals_file, 'w') # MDTMP
        print >>f, txt
        print libxc_functionals_file + ' generated'
        f.close()

if __name__ == '__main__':
    from libxc import Libxc
#    Libxc.construct_libxc_functionals_dict(Libxc())
    Libxc.construct_libxc_functionals_dict(Libxc(), 'c/libxc/src/xc_funcs.h')
