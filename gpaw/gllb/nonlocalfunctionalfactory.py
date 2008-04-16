class NonLocalFunctionalFactory:
    """Factory class.

    NonLocalFunctionalFactory is used by xc_functional.py, when the
    functional starts with words GLLB.
    
    It contains a method called get_functional_by_name, which takes
    the xc-name for non-local functional and returns the corresponding
    GLLBFunctional object. 

    * GLLB (The fermi-level reference set to HOMO)
    * GLLBplusC (GLB with PW91 correlation scr-potential) 
    * GLLBexp (Something to play with)
    """

    def get_functional_by_name(self, name):
        print "Functional name", name
        K_G = 0.382106112167171
        KC_G = 0.470
        from gpaw.gllb.gllb import GLLBFunctional
        from gpaw.gllb.saop import SAOPFunctional
        if name == 'GLLB':
            return GLLBFunctional('X_B88-None',None, K_G)
        elif name == 'GLLBLDA':
            return GLLBFunctional('None-None','LDA', 0)
        elif name == 'GLLBLDARCR':
            return GLLBFunctional('None-None','LDA', 0, relaxed_core_response=True)
        elif name == 'GLLBplusC':
            return GLLBFunctional('X_B88-C_PW91',None,KC_G)
        elif name == 'GLLBRCR':
             return GLLBFunctional('X_B88-None',None, K_G, relaxed_core_response=True)
        elif name == 'GLLBplusCRCR':
             return GLLBFunctional('X_B88-C_PW91',None, KC_G, relaxed_core_response=True)
        elif name == 'SAOP':
             return SAOPFunctional('GLLB', 'LBalpha')
        else:
            raise RuntimeError('Unkown NonLocal density functional: ' + name)

