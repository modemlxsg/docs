import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon

rect = [[266.12210845947266, 25.772997856140137], [343.92846816778183, 25.772997856140137], [343.92846816778183, 41.00000155199166], [266.12210845947266, 41.00000155199166]]

poly = Polygon(rect)

poly = shapely.affinity.rotate(poly, 45.1, (0, 0))

poly_np = np.array(poly)

print(poly_np)