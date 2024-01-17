import numpy as np
import copy
from matplotlib import pyplot as plt

#creating graph from nodes
def distance(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

node_0 = (-1, 1)
node_1 = (0, 9)
node_2 = (1, 2)
node_3 = (4, 5)
node_4 = (10,2)
node_5 = (17, 21)
node_6 = (15, 10)
node_7 = (13, 15)
node_8 = (1, 19)
nodes = [node_0, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8]

adj = [[0, 0, 0, 0, 1, 0, 0, 0, 1],
       [0, 0, 1, 1, 0, 0, 0, 0, 1],
       [0, 1, 0, 1, 1, 0, 0, 0, 0],
       [0, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 1, 1, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 1],
       [0, 0, 0, 0, 1, 1, 0, 1, 0],
       [0, 0, 0, 1, 0, 1, 1, 0, 1],
       [1, 1, 0, 1, 0, 1, 0, 1, 0]]


pts = [[0 for i in range(len(nodes))] for i in range(len(nodes))]
for i in range(len(nodes)):
    for j in range(len(nodes)):
        pts[i][j] = distance(nodes[i], nodes[j])

# running dijkstra weighting
# Credit: geeksforgeeks
class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])

    def minDistance(self, dist, sptSet):
        min = 1e7
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
        return min_index

    def dijkstra(self, src):
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True
            for v in range(self.V):
                if (self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
        return dist

#running laplacian counting
def laplacian(adj, maxPaths):
    for i in range(maxPaths-1):
        adj += np.matmul(adj, adj)
    return adj

def summation(matrix):
    return [np.sum(i) for i in matrix]

# Gift Wrapping
def leftmost(points):
    minim = 0
    for i in range(1,len(points)):
        if points[i][0] < points[minim][0]:
            minim = i
        elif points[i][0] == points[minim][0]:
            if points[i][1] > points[minim][1]:
                minim = i
    return minim

def det(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) -(p2[1] - p1[1]) * (p3[0] - p1[0])

def giftWrapping(nodes, scores):
    hull = []
    l = leftmost(nodes)
    leftMost = nodes[l]
    s = nodes.pop(l)
    t = scores.pop(l)
    nodes.insert(0, s)
    scores.insert(0, l)
    currentVertex = leftMost
    hull.append(currentVertex)
    nextVertex = nodes[1]
    index = 2
    nextIndex = -1
    while True:
        checking = nodes[index]
    
        crossProduct = det(currentVertex, nextVertex, checking)
        if crossProduct < 0:
            nextVertex = checking
        index += 1
        if index == len(nodes):
            if nextVertex == leftMost:
                break
            index = 0
            hull.append(nextVertex)
            currentVertex = nextVertex
            nextVertex = leftMost
    return hull

def hullCheck(nodes, scores, selectedNode, tot):
    hull = giftWrapping(copy.deepcopy(nodes), copy.deepcopy(scores))
    if len(hull) < tot:
        return None
    else:
        index = scores.index(min(scores))
        a = nodes.pop(index)
        b = scores.pop(index)
        if hullCheck(nodes, scores, selectedNode, tot) is None:
            nodes.insert(index, a)
            scores.insert(index, b)
        return giftWrapping(nodes, scores)

def scatter_plot(adj, coords, convex_hull = None):
    xs, ys = zip(*coords) #unzip into x and y coordinates
    plt.scatter(xs, ys)

    for i in range(len(adj)):
        for j in range(len(adj[i])):
            if adj[i][j] == 1:
                c0 = coords[i]
                c1 = coords[j]
                plt.plot((c0[0], c1[0]), (c0[1], c1[1]), 'b')
    
    if convex_hull:
        for i in range(1, len(convex_hull) + 1):
            if i == len(convex_hull): i = 0 #wrap
            c0 = convex_hull[i-1]
            c1 = convex_hull[i]
            plt.plot((c0[0], c1[0]), (c0[1], c1[1]), 'r')
    plt.show()

# Driver program
def main():
    maxPaths = 4
    selectionNode = 3

    g = Graph(len(nodes))
    g.graph = pts
 
    dist = g.dijkstra(selectionNode)
    pathsPerNode = summation(laplacian(adj, maxPaths))
    scores = [float(pathsPerNode[i])/sum(pathsPerNode) - float(dist[i])/sum(dist) for i in range(len(nodes))]

    nodesMut = copy.deepcopy(nodes)
    scoreMut = copy.deepcopy(scores)
    hull = hullCheck(nodesMut, scoreMut, nodes[selectionNode], sum(adj[selectionNode])-1)    
    scatter_plot(adj, nodes, hull)

main()