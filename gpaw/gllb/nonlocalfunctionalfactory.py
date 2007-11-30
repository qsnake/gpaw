class NonLocalFunctionalFactory:
    """Factory class.

    NonLocalFunctionalFactory is used by xc_functional.py, when the
    functional starts with words GLLB.
    
    It contains a method called get_functional_by_name, which takes
    the xc-name for non-local functional and returns the corresponding
    XCNonLocalFunctional object. Currently there are 5 keywords
    available: 
       GLLB (The fermi-level reference set to HOMO)
       GLLBLUMO (The fermi-level reference set to LUMO)
       GLLBRC (GLLB with relaxed core response weights)
       GLLBplusC (GLLB with P86 correlation potential)
       GLLBplusCLUMO (GLLBplusC + fermi-level reference at LUMO)
       GLLBSlaterCore (Currently disabled)
       KLI (Currently disabled)
    """

    def get_functional_by_name(self, name):
        print "Functional name", name
        if name == 'GLLB':
            from gpaw.gllb.gllb import GLLBFunctional
            return GLLBFunctional()
        elif name == 'GLLBRC':
            from gpaw.gllb.gllb import GLLBFunctional
            return GLLBFunctional(relaxed_core_response=True)
        elif name == 'GLLBplusC':
            from gpaw.gllb.gllbc import GLLBCFunctional
            return GLLBCFunctional()
        elif name == 'GLLBLUMO':
            from gpaw.gllb.gllb import GLLBFunctional
            return GLLBFunctional(lumo=True)
        elif name == 'GLLBRCLUMO':
            from gpaw.gllb.gllb import GLLBFunctional
            return GLLBFunctional(lumo=True, relaxed_core_response=True)
        elif name == 'GLLBplusCLUMO':
            from gpaw.gllb.gllbc import GLLBCFunctional
            return GLLBCFunctional(lumo=True)
        #elif name == 'GLLBSlaterCore':
        #    from gpaw.gllb.gllbsc import GLLBSlaterCoreFunctional
        #    return GLLBSlaterCoreFunctional()
        #elif name == 'KLI':
        #    from gpaw.gllb.kli import KLIFunctional
        #    return KLIFunctional()
        else:
            raise RuntimeError('Unkown NonLocal density functional: ' + name)

