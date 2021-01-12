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
#H = nx.from_edgelist([(0, 1), (1, 2), (0, 2), (1, 3)])
#pos = nx.spring_layout(H, iterations=200)

# here goes your statuses as a list of lists
statuses = [[0, 1, 2, 3, 4], [1, 4, 2, 3, 0], [4, 2, 3, 0, 1]]
colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'black', 4:'yellow'}

activations = []

activation1 = np.random.uniform(0, 1, 3)
activations.append(np.concatenate([[1, 1], activation1]))
activation2 = np.random.uniform(0, 1, 3)
activations.append(np.concatenate([[1, 1], activation2]))
activation3 = np.random.uniform(0, 1, 3)
activations.append(np.concatenate([[1, 1], activation3]))

def color_mapper(x):
    c_map = []
    for i in x:
        if i == 1:
            c_map.append('blue')
        elif  i >= 0.75:
            c_map.append('#fff352')
        elif i >= 0.5 and i < 0.75:
            c_map.append('#fcf26a')
        elif i >= 0.25 and i < 0.5:
            c_map.append('#f7f199')
        else:
            c_map.append('#fffde0')
    return c_map

#for a in activations:
    #print(color_mapper(a))
# create a generator from your statuses
# which yields the corresponding color map for each new status
def status():
    for a in activations:
        yield color_mapper(a)  # map statuses to their colors

color_map = status()



def draw_next_status(n):
    plt.cla()
    c_map = next(color_map)
    nx.draw(G, nodes, node_color=c_map,  with_labels=True)


ani = animation.FuncAnimation(plt.gcf(), draw_next_status, interval=1000, frames=3, repeat=False)

plt.show()