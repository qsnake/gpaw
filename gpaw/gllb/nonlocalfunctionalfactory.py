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
        elif name == 'GLLBexp':
            return GLLBFunctional('X_LDA-None','None-C_VWN', KC_G)
        elif name == 'GLLBplusC':
            return GLLBFunctional('X_B88-C_PW91',None,KC_G)
        elif name == 'SAOP':
             return SAOPFunctional('GLLB', 'LBalpha')
        else:
            raise RuntimeError('Unkown NonLocal density functional: ' + name)

