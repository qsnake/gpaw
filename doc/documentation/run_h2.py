# creates: h2.txt
import os
assert os.system('python h2.py') == 0
os.system('cp h2.txt ../_build')
