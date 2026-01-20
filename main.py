from math import pi, cos, sin, sqrt
import matplotlib.pyplot as plt
from quad_pull import quad_extract
from bes_push import add_fcu_beziers_after_quad_fp_poly

def coil_points_gen(r0,n):
    m = 1/cos(pi/8)
    thetas = [i*pi/4 for i in range(n*8+1)]
    thetasbes = [i*pi/4+pi/8 for i in range(n*8)]
    main_coil_points = []
    for theta in thetas:
        main_coil_points.append(spiral(theta, r0, n))
    bes_coil_points = []
    for theta in thetasbes:
        bes_coil_points.append(spiral(theta, r0, n, m))
    besier_handles = []
    for i in range(len(bes_coil_points)):
        besier_handles.append(average_points(main_coil_points[i], bes_coil_points[i]))
        besier_handles.append(average_points(main_coil_points[i+1], bes_coil_points[i]))
    points = []
    for i in range(len(main_coil_points)-1):
        points.append(apply_delta(main_coil_points[i]))
        points.append(apply_delta(besier_handles[2*i]))
        points.append(apply_delta(besier_handles[2*i+1]))
        points.append(apply_delta(main_coil_points[i+1]))
    points = [rotate(point) for point in points]
    return normalise(points)

def spiral(theta,r0,n, m=1):
    r = r0 + (1-r0)*theta/(2*pi*n)
    delta = r*0.06*cos(4*theta)
    r*=m
    return cos(theta)*r, sin(theta)*r, cos(theta)*delta, sin(theta)*delta

def apply_delta(point):
    return point[0]+point[2], point[1]+point[2]

def average_points(point1, point2):
    return (point1[0]+point2[0])/2, (point1[1]+point2[1])/2, point1[2], point1[3]

def normalise(points):
    return [(point[0]/2 + 0.5, point[1]/2 + 0.5) for point in points]

def lerp(p1, p2, p3, p4, p):
    p_x = p1[0]*(1-p[0])*(1-p[1]) + p2[0]*(1-p[0])*(p[1]) + p3[0]*(p[0])*(p[1]) + p4[0]*(p[0])*(1-p[1])
    p_y = p1[1]*(1-p[0])*(1-p[1]) + p2[1]*(1-p[0])*(p[1]) + p3[1]*(p[0])*(p[1]) + p4[1]*(p[0])*(1-p[1])
    
    return p_x, p_y

def lf(x):
    return cos(pi*x)/2 + 0.5

def rotate(point):
    return 1/sqrt(2)*point[0]-1/sqrt(2)*point[1], 1/sqrt(2)*point[0]+1/sqrt(2)*point[1]
    
path = "footprints.pretty/coil.kicad_mod"
outpath = "footprints.pretty/coil_done.kicad_mod"
lerp_points = quad_extract(path)

points = coil_points_gen(0.3,5)
lpoints = []
# p1 = (-1.5,0)
# p2 = (-1,1)
# p3 = (1,1)
# p4 = (1.5,0)
# p1 = (-1,0)
# p2 = (-1,1)
# p3 = (1,1)
# p4 = (1,0)

for point in points:
    lpoints.append(lerp(*lerp_points,point))


add_fcu_beziers_after_quad_fp_poly(0.3, lpoints, path, outpath)


# plt.plot([point[0] for point in points], [point[1] for point in points], marker="o")
plt.plot([point[0] for point in lpoints], [point[1] for point in lpoints], marker="o")
plt.show()
