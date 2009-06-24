from gpaw.pes.folding import folding_routine

class BasePES:
    def __init__(self):
        self.h=1

    def save_folded_pes(self,
             filename=None,
             width=0.5, # Gauss/Lorentz width
             emin=None,
             emax=None,
             de=None,
             folding='Gauss',
             comment=None):

        folding_routine(self.get_energies_and_weights(),width,filename,
                emin,emax,de,folding,comment)

    def get_energies_and_weights(self):
        if self.be == None or self.f == None:
            self._calculate()

        return self.be , self.f

