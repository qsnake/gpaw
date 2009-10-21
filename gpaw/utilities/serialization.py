import sys
    
from optparse import OptionParser, OptionGroup
from gpaw.parameters import InputParameters
from gpaw.poisson import PoissonSolver
from gpaw.mixer import Mixer, MixerSum, MixerDif
from gpaw import GPAW

# Problems and unresolved issues
# ------------------------------
#
# what do we do about large objects such as PoissonSolvers?
# option parsers have cyclic references, maybe call destroy()


def build_parser():
    usage = '%prog [OPTIONS] [FILE]'
    description = ('Print representation of GPAW input parameters to stdout.')
    parser = OptionParser(usage=usage, description=description)
    g = OptionGroup(parser, 'General')
    g.add_option('--complete', action='store_true', default=False,
                 help='print complete set of input parameters')
    g.add_option('--pretty', action='store_true', default=False,
                 help='format output nicely')
    #g.add_option('--validate', action='store_true', default=False,
    #             help='validate arguments to some extent')
    parser.add_option_group(g)
    return parser


def append_to_optiongroup(parameters, opts):
    for key, value in parameters.items():
        opts.add_option('--%s' % key, default=repr(value), type=str,
                        help='default=%default')
    return opts


def deserialize(filename):
    """Get an InputParameters object from a filename."""
    stringrep = open(filename).read()
    if not stringrep.startswith('InputParameters(**{'):
        raise ValueError('Does not appear to be a serialized InputParameters')
    parameters = eval(stringrep)
    return parameters


def populate_parser(parser, defaultparameters):
    opts = OptionGroup(parser, 'GPAW parameters')
    append_to_optiongroup(defaultparameters, opts)
    parser.add_option_group(opts)
    return parser


def main(argv):
    # build from scratch
    # build from existing
    # build from gpw?
    # just print nicely?
    # print as python script?
    defaults = InputParameters()
    parser = build_parser()
    populate_parser(parser, defaults)

    opts, args = parser.parse_args(argv)
    
    for arg in args:
        deserialized_parameters = deserialize(args[0])
        # We have to use the newly loaded info (target file)
        # to get new defaults!
        #
        # Rather ugly but hopefully it works
        # We can probably avoid this somehow, think about it in the future
        # (can one have different call formats like e.g. the 'du' man-page,
        # and somehow detect which one is relevant?)
        parser2 = build_parser()
        populate_parser(parser2, deserialized_parameters)
        opts, args2 = parser2.parse_args(argv)

    parameters = {}
    for key, value in vars(opts).items():
        if key in defaults:
            parameters[key] = eval(value)
    output = InputParameters(**parameters)

    # Remove keys which are not changed from defaults
    if not opts.complete:
        for key in defaults:
            if defaults[key] == output[key]:
                # XXX this is probably not meant to be done
                del output[key]
    
    if opts.pretty:
        start = 'InputParameters('
        indent = ' ' * len(start)
        keyvals = ['%s=%s' % (key, repr(value))
                   for key, value in output.items()]
        end = ')\n'

        sys.stdout.write(start)
        
        keyval_string = (',\n%s' % indent).join(keyvals)
        sys.stdout.write(keyval_string)
        sys.stdout.write(end)
        
    else:
        print output

    #if opts.validate:
    #    output['txt'] = None
    #    calc = GPAW(**output)
