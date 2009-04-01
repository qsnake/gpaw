# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""db archiving module.

"""

import os

# reads the path for the repository from 
# the environment variable DB_REPOSITORY
# and returns "", if not set
def get_db_path():
    path = os.getenv('DB_REPOSITORY')
    if path == None or len(path.strip())==0:
       return ""
    else:
       return path.strip()
