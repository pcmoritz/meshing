import numpy
import networkx as nx

# The grid we draw is N by N by N, each field is w wide
N = 5
w = 0.1
input_file = open('~/function-values.txt', 'r')

# read in the file
content = input_file.readlines()

def index(x, y, z):
    return x * N * N + y * N + z

from mayavi import mlab
mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
mlab.clf()

X = numpy.zeros(N * N * N)
Y = numpy.zeros(N * N * N)
Z = numpy.zeros(N * N * N)
S = numpy.ones(N * N * N)

# connection graph:
G = nx.Graph()

# set nodes appropriately for the "unit map" grid
# for x in range(N):
#     for y in range(N):
#         for z in range(N):
#             X[index(x, y, z)] = x * w
#             Y[index(x, y, z)] = y * w
#             Z[index(x, y, z)] = z * w

index = 0

# read in the grid from the file
for entry in content:
    if entry == "NA":
        X[index] = float('NaN')
        Y[index] = float('NaN')
        Z[index] = float('NaN')
    else:
        X[index] = entry[0]
        X[index] = entry[1]
        X[index] = entry[2]
    index = index + 1

# connect the graph
for x in range(N):
    for y in range(N):
        for z in range(N):
            if X[index(x, y, z)] == float('NaN'):
                continue
            
            if not x == N-1 and not X[index(x+1, y, z)] != float('NaN'):
                G.add_edge(index(x, y, z), index(x+1, y, z))
            if not y == N-1 and not X[index(x, y+1, z)] != float('NaN'):
                G.add_edge(index(x, y, z), index(x, y+1, z))
            if not z == N-1 and not X[index(x, y, z+1)] != float('NaN'):
                G.add_edge(index(x, y, z), index(x, y, z+1))

node_size=0.005
edge_color=(0.8, 0.8, 0.8)
edge_size=0.001

nodes = sorted(G.nodes())
            
pts = mlab.points3d(X[nodes], Y[nodes], Z[nodes], S[nodes], 
                    scale_factor=node_size)
pts.mlab_source.dataset.lines = numpy.array(G.edges())
tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
mlab.pipeline.surface(tube, color=edge_color)

mlab.show()
