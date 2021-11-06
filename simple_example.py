from math import pi

from src.simulation import Simulation
from src.control import GeometricController
from src.plant import RadmacherPlant
plant = RadmacherPlant('parafoil.yaml')
controller = GeometricController(plant)
s = Simulation(plant, controller)

x0 = [5, 0, pi/2, -500, 100, 500, 0, 0]
t, y = s.run_to_time(0.1, 1000, x0, plot_result=True, generate_gif=True)
