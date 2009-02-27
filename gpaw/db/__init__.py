# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""db archiving module.

"""

#defines where to automatically store a copy of the output
global db_path
db_path = "/home/niflheim/dlandis/database"


def set_db_path(path): 
    global db_path
    db_path = path

def get_db_path():
    global db_path
    return db_path
