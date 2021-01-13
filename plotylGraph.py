'''
import plotly.graph_objects as go

import networkx as nx
import matplotlib.pyplot as plt
# Create a graph object
G = nx.MultiGraph()

#Add the nodes


Initialise nodes and position

#Initialise
layers = [2, 2, 1]
nodes = {'input1': (10, 20), 'input2':(10, 40), 'a1': (20, 20), 'a2': (20, 40), 'a3': (30, 30)}

#Add nodes to the Graph
G.add_nodes_from(nodes.keys(), color='r')

#Include the positions
for n, p in nodes.items():
    G.nodes[n]['pos'] = p

#Add the edges
G.add_edge('input1', 'a1', color='r', weight = 2)
G.add_edge('input1', 'a2', color='r',  weight = 2)
G.add_edge('input2', 'a1', color='r',  weight = 4)
G.add_edge('input2', 'a2', color='r',  weight = 4)
G.add_edge('a1', 'a3', color='r',  weight = 6)
G.add_edge('a2', 'a3', color='r',  weight = 6)

edges = G.edges()
colors = [G[u][v][0]['color'] for u,v in edges]
weights = [G[u][v][0]['weight'] for u,v in edges]

nx.draw(G, pos = nodes, edge_color=colors, width=weights, with_labels=True)
plt.show()

total_weights = 0

for i in range(len(layers) - 1):
    total_weights += layers[i + 1] * layers[i]

n = 0
for i in layers:
    for j in range(i):
        print(n)
    n += 1





G.add_node('input1', )
G.add_node('input2')
G.add_node('a1')
G.add_node('a2')
G.add_node('a3')

#Add edges

G.add_edge('input1', 'a1', color='r', weight = 2)
G.add_edge('input1', 'a2', color='r',  weight = 2)
G.add_edge('input2', 'a1', color='r',  weight = 4)
G.add_edge('input2', 'a2', color='r',  weight = 4)
G.add_edge('a1', 'a3', color='r',  weight = 6)
G.add_edge('a2', 'a3', color='r',  weight = 6)

edges = G.edges()
colors = [G[u][v][0]['color'] for u,v in edges]
weights = [G[u][v][0]['weight'] for u,v in edges]

#Visualise using matplotlib
plt.subplot(121)
nx.draw(G, edge_color=colors, width=weights, with_labels=True)
plt.show()

for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    print(x0, y0)
    
G = nx.random_geometric_graph(200, 0.125)
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

print(edge_x)
'''

import matplotlib
matplotlib.use('TKAgg')
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

layers = [2, 2, 1]
nodes = {'input1': (10, 20), 'input2':(10, 40), 'a1': (20, 20), 'a2': (20, 40), 'a3': (30, 30)}
G = nx.MultiGraph()
G.add_nodes_from(nodes.keys(), color='r')
#Include the positions
for n, p in nodes.items():
    G.nodes[n]['pos'] = p

G.add_edge('input1', 'a1', color='r', weight = 2)
G.add_edge('input1', 'a2', color='r',  weight = 2)
G.add_edge('input2', 'a1', color='r',  weight = 4)
G.add_edge('input2', 'a2', color='r',  weight = 4)
G.add_edge('a1', 'a3', color='r',  weight = 6)
G.add_edge('a2', 'a3', color='r',  weight = 6)

activations = []

activation1 = np.random.uniform(0, 1, 3)
activations.append(np.concatenate([[1, 1], activation1]))
activation2 = np.random.uniform(0, 1, 3)
activations.append(np.concatenate([[1, 1], activation2]))
activation3 = np.random.uniform(0, 1, 3)
activations.append(np.concatenate([[1, 1], activation3]))

weights = [[0.6130479949441281, -0.728200815192963, 0.4554891935445306, 0.3333301108489286, 0.5494843570230445, 0.43645197694509197, 0.5552372592273038, 0.4368136871662169, 0.7030286374472023],
           [0.2130479949441281, -0.928200815192963, 0.7554891935445306, 0.6333301108489286, 0.5494843570230445, 0.63645197694509197, 0.9552372592273038, 0.9368136871662169, 0.5030286374472023],
           [0.5130479949441281, -0.928200815192963, 0.2554891935445306, 0.9333301108489286, 0.5494843570230445, 0.53645197694509197, 0.5552372592273038, 0.4368136871662169, 0.7030286374472023]]
def color_mapper_node(x):
    c_map = []
    for i in x:
        if i == 1:
            c_map.append('blue')
        elif  i >= 0.75:
            c_map.append('#dffc00')
        elif i >= 0.5 and i < 0.75:
            c_map.append('#c6d934')
        elif i >= 0.25 and i < 0.5:
            c_map.append('#c8d65c')
        else:
            c_map.append('#afb86a')
    return c_map

def color_mapper_edge(x):
    edge_map = []
    for i in x:
        i = np.abs(i)
        if i >= 0.70:
            edge_map.append('#ff0015')
        elif i >= 0.5 and i < 0.75:
            edge_map.append('#ff5e6c')
        elif i >= 0.25 and i < 0.5:
            edge_map.append('#fa848e')
        else:
            edge_map.append('#ff9ca4')
    return edge_map

#for a in activations:
    #print(color_mapper(a))
# create a generator from your statuses
# which yields the corresponding color map for each new status
def status():
    for a in activations:
        yield color_mapper_node(a)  # map statuses to their colors

def status_edge():
    for w in weights:
        yield color_mapper_edge(w)  # map statuses to their colors

color_map = status()
edge_map = status_edge()



def draw_next_status(n):
    plt.cla()
    c_map = next(color_map)
    e_map = next(edge_map)
    nx.draw(G, nodes, node_color=c_map, edge_color=e_map, width = 4, with_labels=True)


ani = animation.FuncAnimation(plt.gcf(), draw_next_status, interval=1000, frames=5, repeat= False)

plt.show()