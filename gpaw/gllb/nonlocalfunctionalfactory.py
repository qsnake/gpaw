class NonLocalFunctionalFactory:
    """Factory class.

    NonLocalFunctionalFactory is used by xc_functional.py, when the
    functional starts with words GLLB.
    
    It contains a method called get_functional_by_name, which takes
    the xc-name for non-local functional and returns the corresponding
    NonLocalFunctional object. 

    * GLLB
    * GLLBLDA (A test functional, which is just LDA but via
               NonLocalFunctional framework)
    """

    def get_functional_by_name(self, name):
        print "Functional name", name
        K_G = 0.382106112167171
        KC_G = 0.470

        from gpaw.gllb.nonlocalfunctional import NonLocalFunctional
        functional = NonLocalFunctional()
        
        if name == 'GLLB':
            # Functional GLLB
            # Contains screening part from GGA functional
            # And response part based on simple square root expection
            # of orbital energy differences.
            from gpaw.gllb.contributions.c_gllbscr import C_GLLBScr
            from gpaw.gllb.contributions.c_response import C_Response
            C_Response(functional, 1.0,
                       C_GLLBScr(functional, 1.0).get_coefficient_calculator())
            return functional
        elif name == 'GLLBC':
                        from gpaw.gllb.contributions.c_gllbscr import C_GLLBScr
                        from gpaw.gllb.contributions.c_response import C_Response
                        from gpaw.gllb.contributions.c_lda import C_LDA
                        C_Response(functional, 1.0,
                        C_GLLBScr(functional, 1.0,'PBE').get_coefficient_calculator())
                        #C_LDA(functional, 1.0, 'LDA')
                        #C_LDA(functional, -1.0, 'LDAx')
                        return functional
                                                        
        elif name == 'GLLBLDA':
            from gpaw.gllb.contributions.c_lda import C_LDA
            C_LDA(functional, 1.0)
            return functional
        elif name == 'GLLBSLATER':
            from gpaw.gllb.contributions.c_slater import C_Slater
            C_Slater(functional, 1.0)
            return functional
        elif name == 'GLLBNORESP':
            from gpaw.gllb.contributions.c_gllbscr import C_GLLBScr
            C_GLLBScr(functional, 1.0)
            return functional
        elif name == 'KLI':
            raise RuntimeError('KLI functional not implemented')
            from gpaw.gllb.contributions.c_slater import C_Slater
            from gpaw.gllb.contributions.c_response import C_Response
            C_Response(functional, 1.0,
                       C_Slater(functional, 1.0).get_coefficient_calculator())
            return functional
        else:
            raise RuntimeError('Unkown NonLocal density functional: ' + name)

