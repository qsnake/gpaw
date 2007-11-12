class NonLocalFunctionalFactory:
    """Factory class.

    NonLocalFunctionalFactory is used by xc_functional.py, when the
    functional starts with words NL.
    
    It contains a method called get_functional_by_name, which takes
    the xc-name for non-local functional and returns the corresponding
    XCNonLocalFunctional object. Currently there are 3 keywords
    available: GLLB GLLBSlaterCore GLLBKLI

    """

    def get_functional_by_name(self, name):
        if name == 'GLLB':
            from gpaw.gllb.gllb import GLLBFunctional
            return GLLBFunctional()
        elif name == 'GLLBSlaterCore':
            from gpaw.gllb.gllbsc import GLLBSlaterCoreFunctional
            return GLLBSlaterCoreFunctional()
        elif name == 'KLI':
            from gpaw.gllb.kli import KLIFunctional
            return KLIFunctional()
        else:
            raise RuntimeError('Unkown NonLocal density functional: ' + name)

