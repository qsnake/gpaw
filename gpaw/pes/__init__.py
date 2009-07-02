import gpaw.mpi as mpi
from gpaw.lrtddft.spectrum import Writer

class PESpectrum(Writer):
    def __init__(self, 
                 enlist,
                 folding='Gauss', 
                 width=0.08 # Gauss/Lorentz width
                 ):
        Writer.__init__(self, folding, width)
        self.title = 'Photo emission spectrum'
        self.fields = 'Binding energy [eV]     Folded spectroscopic factor'
    
        self.energies = enlist[0]
        self.values = []
        for val in enlist[1]:
            self.values.append([val])

class BasePES:
    def save_folded_pes(self,
             filename=None,
             width=0.5, # Gauss/Lorentz width
             emin=None,
             emax=None,
             de=None,
             folding='Gauss',
             comment=None):

        ew = self.get_energies_and_weights()
        if mpi.rank == mpi.MASTER:
            sp = PESpectrum(ew, folding, width)
            sp.write(filename, emin, emax, de,  comment)

    def get_energies_and_weights(self):
        if self.be == None or self.f == None:
            self._calculate()

        return self.be , self.f

    def set_first_peak_energy(self,energy):
        self.first_peak_energy=energy
        

