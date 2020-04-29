def constant_controller(x):
    """Dummy controller function. Returns zero commands for both inputs to the parafoil. Ordinarily you would do some calculations with x to produce the commands u.

    Arguments:
        x {iterable} -- the state of the parafoil

    Returns:
        list -- list of zeros for the control.
    """
    return [0.1, 0.1]


def minimum_time_controller(x, a, b):
    """Implemented from Rademarcher (2009) section 4.2. Uses simplified dynamics to solve for minimum time trajectories consisting of straight lines over ground and maximum turn rate circular arcs.

    Arguments:
        x {[type]} -- [description]
    """

    p_psi = 0

    if p_psi == 0:
        u = 0
    elif p_psi > 0:
        u = a
    else:
        u = b

    return [u, 0]
