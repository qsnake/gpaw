
class Perturbation:
    """Abstract base class for various perturbation types.

    For the moment no specific interface is implemented.

    """
    
    def __init__(self):
        pass

    def get_dtype(self):
        """Return dtype for the perturbation."""
        
        raise NotImplementedError, ("Implement in derived classes")
