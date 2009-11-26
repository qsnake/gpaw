# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""db archiving module (deprecated)."""

print "gpaw.db:"
print "========"
print ""
print "gpaw.db is outdated and was replaced with gpaw.cmr."
print ""
print "gpaw.cmr is part of the Computational Materials Repository (CMR)"
print "and allows you to put the results of your calculation in"
print "a database from where you can retreive (sort/filter) the results again."
print ""
print "In order to use the Computational Materials Repository (CMR) code"
print "do the following:"
print "  1. mkdir -p $HOME/lib/python/cmr"
print "  2. svn co https://svn.fysik.dtu.dk/projects/cmr/trunk $HOME/lib/python/cmr"
print "  3a. add $HOME/lib/python/cmr to your .bashrc:"
print "        export PYTHONPATH=\"$HOME/lib/python/cmr:$PYTHONPATH\""
print "  3b. add $HOME/lib/python/cmr to your .tcshrc:"
print "        setenv PYTHONPATH $HOME/lib/python/cmr:$PYTHONPATH"


import os

# reads the path for the repository from 
# the environment variable CMR_REPOSITORY
# and returns "", if not set
def get_db_path():
    path = os.getenv('CMR_REPOSITORY')
    if path == None or len(path.strip())==0:
       return ""
    else:
       return path.strip()
