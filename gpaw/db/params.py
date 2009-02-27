#! /usr/bin/python

class params:
    """Base class for parameter files. Please
       use derived classes in your code.
       
       Derived classes: gpaw_params
                          dacapo_params
    """
    
    

    # self.data: a dictionary (see derived classes for examples)
    def __init__(self, data):
        self.data = data 
        # the inverse dictionary with the output names (xml names)
        # as keys and the local names as values
        self.data_inv={}
        for k in self.data.keys():
            self.data_inv[self.data[k]["xml_name"]] = {"local_name":k}

    def get_inv(self, item):
        if self.data_inv.has_key(item):
           return self.data_inv[item]
        else:
           return {"local_name":item, "python_type":None}

    def get(self, item):
        if self.data.has_key(item):
           return self.data[item]
        else:
           return {"xml_name":item, "python_type":None}

    def has_key(self, key):
        return self.data.has_key(key)

    def __getitem__(self, name):
        return self.data[name]
    
