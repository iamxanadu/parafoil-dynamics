def constant_controller(x):
    """Dummy controller function. Returns zero commands for both inputs to the parafoil. Ordinarily you would do some calculations with x to produce the commands u.
    
    Arguments:
        x {iterable} -- the state of the parafoil
    
    Returns:
        list -- list of zeros for the control.
    """
    return [0, 0]