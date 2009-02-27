# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""db archiving module.

"""

import os

# defines where to automatically store a copy of the output
# if no environment variables is set
global db_path
db_path = "/home/niflheim/repository/db"


def set_db_path(path): 
    global db_path
    db_path = path

def get_db_path():
    path = os.getenv('DB_REPOSITORY')
    if path ==None or len(path.strip())==0:
       global db_path
       return db_path
    else:
       return path.strip()
