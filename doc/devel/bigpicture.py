# creates: bigpicture.pdf bigpicture.png
import os
from math import pi, cos, sin

import numpy as np

latex = r"""\documentclass[10pt,landscape]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\parindent=0pt
\pagestyle{empty}
%\usepackage{landscape}
\usepackage{pstricks-add,graphicx,hyperref}
\usepackage[margin=1cm]{geometry}
\newsavebox\PSTBox
%\special{papersize=420mm,297mm}
\begin{document}

\psset{framesep=2mm,arrowscale=1.75}

\begin{pspicture}(0,0)(27,18)
\psframe*[linecolor=green!15](21.5,13)(26,17.9)
\rput(22,17.5){ASE}
%\newrgbcolor{yellow7}{0.97 0.5 0.85}
"""

all = []
names = {}

class Box:
    def __init__(self, name, description=None, attributes=None,
                 color='black!20', width=None):
        self.position = None
        self.name = name
        if isinstance(description, str):
                description = [description]
        self.description = description
        self.attributes = attributes
        self.width = width
        self.color = color
        self.owns = []
        all.append(self)
        if name in names:
            names[name] += 1
            self.id = name + str(names[name])
        else:
            self.id = name
            names[name] = 1
            
    def set_position(self, position):
        self.position = np.asarray(position)

    def to_latex(self):
        if self.width:
            format = 'p{%fcm}' % self.width
        else:
            format = 'c'
        boxes = [
            '\\rput(%f,%f){' % tuple(self.position) +
            '\\rnode{%s}{' % self.id +
            '\\psshadowbox[fillcolor=%s,fillstyle=solid]{' %
            self.color + '\\begin{tabular}{%s}' % format]
        url = '\\href{https://wiki.fysik.dtu.dk/gpaw/'
        table = [url + 'devel/devel.html}{\small %s}' % self.name]
        if self.description:
            table.extend(['{\\tiny %s}' % txt for txt in self.description])
        if self.attributes:
            table.append('{\\tiny \\texttt{%s}}' %
                         ', '.join(self.attributes).replace('_', '\\_'))
        boxes.append('\\\\\n'.join(table))
        boxes += ['\\end{tabular}}}}']
        arrows = []
        for other, name, x in self.owns:
            arrows += ['\\ncline{->}{%s}{%s}' % (self.id, other.id)]
            if name:
                arrows += [
                    '\\rput(%f, %f){\\psframebox*[framesep=0.05]{\\tiny %s}}' %
                    (tuple(((1 - x) * self.position + x * other.position)) +
                     (name.replace('_', '\\_'),))]
                
        return boxes, arrows

    def has(self, other, name, angle=None, distance=None, x=0.55):
        self.owns.append((other, name, x))
        if angle is not None:
            angle *= pi / 180
            other.set_position(self.position +
                               [cos(angle) * distance,
                                sin(angle) * distance])

atoms = Box('Atoms', '', ['positions', 'numbers', 'cell', 'pbc'])
paw = Box('PAW', None, ['initialized'], 'green!70')
scf = Box('SCFLoop', None)
density = Box('Density', 
              [r'$\tilde{n}_\sigma = \sum_{\mathbf{k}n}' +
               r'|\tilde{\psi}_{\sigma\mathbf{k}n}|^2' +
               r'\frac{1}{2}\sum_a \tilde{n}_c^a$',
               r'$\tilde{\rho}(\mathbf{r}) = ' +
               r'\sum_\sigma\tilde{n}_\sigma + \sum_{aL}Q_L^a \hat{g}_L^a$'],
              ['nspins', 'nt_sG', 'nt_sg', 'rhot_g', 'Q_aL', 'D_asp'],
              'green!30')
mixer = Box('Mixer', color='blue!30')
hamiltonian = Box('Hamiltonian',
                  r"""$-\frac{1}{2}\nabla^2 +
 \tilde{v} +
 \sum_a \sum_{i_1i_2} |\tilde{p}_{i_1}^a \rangle 
 \Delta H_{i_1i_2} \langle \tilde{p}_{i_1}^a|$""",
                  ['nspins', 'vt_sG', 'vt_sg', 'vHt_g', 'dH_asp',
                   'Etot', 'Ekin', 'Exc', 'Epot', 'Ebar'])
wfs = Box('WaveFunctions',
          r"""$\tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})$""",
          ['ibzk_qc', 'mynbands'])
gd = Box('GridDescriptor', '(coarse grid)',
         ['cell_cv', 'N_c', 'pbc_c'], 'orange!30')
finegd = Box('GridDescriptor', '(fine grid)',
         ['cell_cv', 'N_c', 'pbc_c'], 'orange!30')
setups = Box('Setups', ['', '', '', ''], ['nvalence', 'nao', 'Eref'],
             width=4.2)
xccorrection = Box('XCCorrection')
nct = Box('LFC', r'$\tilde{n}_c^a(r)$', None, 'red!70')
vbar = Box('LFC', r'$\bar{v}^a(r)$', None, 'red!70')
ghat = Box('LFC', r'$\hat{g}_{\ell m}^a(\mathbf{r})$', None, 'red!70')
grid = Box('GridWaveFunctions',
           r"""$\tilde{\psi}_{\sigma\mathbf{k}n}(ih,jh,kh)$""",
           ['hmm'])
pt = Box('LFC', r'$\tilde{p}_i^a(\mathbf{r})$', None, 'red!70')
lcao = Box('LCAOWaveFunctions',
           r"""$\tilde{\psi}_{\sigma\mathbf{k}n}(\mathbf{r})=
\sum_{\mu\mathbf{R}} C_{\sigma\mathbf{k}n\mu}
\Phi_\mu(\mathbf{r} - \mathbf{R}) \exp(i\mathbf{k}\cdot\mathbf{R})$""",
           ['S_qMM', 'T_qMM', 'P_aqMi'])
atoms0 = Box('Atoms', '(copy)', ['positions', 'numbers', 'cell', 'pbc'])
parameters = Box('InputParameters', None, ['xc', 'nbands', '...'])
forces = Box('ForceCalculator')
occupations = Box(
    'OccupationNumbers',
    r'$\epsilon_{\sigma\mathbf{k}n} \rightarrow f_{\sigma\mathbf{k}n}$')
poisson = Box('PoissonSolver')
eigensolver = Box('EigenSolver')
symmetry = Box('Symmetry')
restrictor = Box('Transformer', '(fine -> coarse)')
interpolator = Box('Transformer', '(coarse -> fine)')
xcfunc = Box('XCFunctional')
xc3dgrid = Box('XC3DGrid')
kin = Box('Operator', r'$-\frac{1}{2}\nabla^2$')
overlap = Box('Overlap')
basisfunctions = Box('BasisFunctions', r'$\Phi_\mu(\mathbf{r})$',
                     color='red!70')
tci = Box('TwoCenterIntegrals',
          r'$\langle\Phi_\mu|\Phi_\nu\rangle,'
          r'\langle\Phi_\mu|\hat{T}|\Phi_\nu\rangle$')

atoms.set_position((23.5, 16))
atoms.has(paw, 'calculator', -160, 7.5)
paw.has(scf, 'scf', 160, 4)
#paw.has(gd, 'gd')
paw.has(density, 'density', -150, 14, 0.23)
paw.has(hamiltonian, 'hamiltonian', 180, 10, 0.3)
paw.has(wfs, 'wfs', -65, 5.5)
paw.has(atoms0, 'atoms', 9, 7.5)
paw.has(parameters, 'input_parameters', 90, 4)
paw.has(forces, 'forces', 50, 4)
paw.has(occupations, 'occupations', 136, 4)
density.has(mixer, 'mixer', 130, 3.3)
density.has(gd, 'gd', x=0.33)
density.has(finegd, 'finegd', 76, 3.5)
density.has(setups, 'setups', 0, 7, 0.45)
density.has(nct, 'nct', -90, 3)
density.has(ghat, 'ghat', -50, 4)
density.has(interpolator, 'interpolator', -135, 4)
hamiltonian.has(restrictor, 'restrictor', 40, 4)
hamiltonian.has(xcfunc, 'xcfunc', -150, 5)
hamiltonian.has(xc3dgrid, 'xc', 130, 4)
hamiltonian.has(vbar, 'vbar', 80, 4)
hamiltonian.has(setups, 'setups', x=0.3)
hamiltonian.has(gd, 'gd')
hamiltonian.has(finegd, 'finegd')
hamiltonian.has(poisson, 'poissonsolver', 150, 6)
wfs.has(gd, 'gd', 159, 5)
wfs.has(setups, 'setups')
wfs.has(grid, None, -55, 6)
wfs.has(lcao, None, -112, 5.2)
wfs.has(eigensolver, 'eigensolver', 30, 4)
wfs.has(symmetry, 'symmetry', 80, 3)
grid.has(pt, 'pt', -45, 4)
grid.has(kin, 'kin', -90, 3)
grid.has(overlap, 'overlap', -135, 3.5)
lcao.has(basisfunctions, 'basis_functions', -145, 4.9)
lcao.has(tci, 'tci', -95, 3.8)

for i in range(3):
    setup = Box('Setup', None,
                ['Z', 'Nv','Nc', 'pt_j','nct', 'vbar','ghat_l'],
                'blue!40', width=2)
    setup.set_position(setups.position +
                       (0.9 - i * 0.14, 0.3 - i * 0.14))
setup.has(xccorrection, 'xc_correction', -110, 3.7)

kpts = [Box('KPoint', None, ['psit_nG', 'C_nM', 'eps_n', 'f_n'])
        for i in range(3)]
wfs.has(kpts[1], 'kpt_u', 0, 5, 0.45)
kpts[0].set_position(kpts[1].position - 0.14)
kpts[2].set_position(kpts[1].position + 0.14)

allboxes = []
allarrows = []
for b in all:
   boxes, arrows = b.to_latex()
   allboxes.extend(boxes)
   allarrows.extend(arrows)
   
latex = [latex] + allboxes + allarrows + ['\\end{pspicture}\n\\end{document}']
open('bigpicture.tex', 'w').write('\n'.join(latex))

os.system('latex bigpicture.tex > bigpicture.log')
os.system('dvipdf bigpicture.dvi')
os.system('cp bigpicture.pdf ../_build')
os.system('convert bigpicture.pdf -resize 50% bigpicture.png')

