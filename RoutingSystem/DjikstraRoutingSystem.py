''' 
Title: Pygame dynamic rerouting simulation based on Dijkstra Algorithm
Author: Jason Thamel
Student ID: 2057941
University of Birmingham 

Description:
This script has been used to design reroutig system based on the Dijkstra Algorithm
'''

import pygame
import pandas as pd
import heapq
import time

#The edges are set to be Bi-directional so that the algorithm can be used for both directions

# Define a Graph class to handle graph operations
class Graph:
    def __init__(self):
        self.nodes = {}

    def addEdge(self, fromNode, toNode, initialTime):
        # Add an edge with the initial weight between two nodes
        if fromNode not in self.nodes:
            self.nodes[fromNode] = {}
        if toNode not in self.nodes:
            self.nodes[toNode] = {}
        self.nodes[fromNode][toNode] = initialTime  # Bi-directional edge
        self.nodes[toNode][fromNode] = initialTime  # Bi-directional edge

    def updateEdge(self, fromNode, toNode, newWeight):
        # Update the weight of an existing edge
        if fromNode in self.nodes and toNode in self.nodes[fromNode]:
            self.nodes[fromNode][toNode] = newWeight # Bi-directional edge
            self.nodes[toNode][fromNode] = newWeight # Bi-directional edge

    def nodeNeighbor(self, node):
        # Get all neighbors of a node
        return self.nodes[node] if node in self.nodes else {}

# Function to execute Dijkstra's algorithm to find the shortest path
def dijkstraAlgo(graph, startNode, endNode):
    distances = {node: float('infinity') for node in graph.nodes}
    distances[startNode] = 0
    priorityQueue = [(0, startNode)]
    previousNodes = {node: None for node in graph.nodes}

    while priorityQueue:
        currentDistance, currentNode = heapq.heappop(priorityQueue)
        if currentNode == endNode:
            break

        for neighbor, weight in graph.nodeNeighbor(currentNode).items():
            distance = currentDistance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previousNodes[neighbor] = currentNode
                heapq.heappush(priorityQueue, (distance, neighbor))

    path = []
    currentNode = endNode
    while currentNode is not None:
        path.append(currentNode)
        currentNode = previousNodes[currentNode]
    path.reverse()
    return path, distances[endNode]


# Function to read the updated edges from a csv file
def csvFileEdges(filename):
    return pd.read_csv(filename)

# Function to apply updates to only specific edges (at crossings)
def update_specific_edges(graph, updates):
    for index, row in updates.iterrows():
        graph.updateEdge(row['fromNode'], row['toNode'], row['newWeight'])

# Initialize Pygame
pygame.init()
width, height = 1280, 720
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Graph Visualization")

backgroundColor = (255, 255, 255)
nodeColor = (230, 145, 50)
edgeColor = (0, 0, 0)
pathColor = (40, 130, 200)
textColor = (0, 0, 0)
font = pygame.font.Font(None, 24)

# Define thee positions of the nodes within the window of the pygame 
# Each node represents a traffic light
nodePositions = {
    'A1': (80, 630), 'A2': (280, 630), 'A3': (320, 630), 'A4': (500, 630), 'A5': (540, 630), 'A6': (760, 630), 'A7': (800, 630),                                       'A10': (1240, 630),
    'B1': (80, 590), 'B2': (280, 590), 'B3': (320, 590), 'B4': (500, 590), 'B5': (540, 590), 'B6': (760, 590), 'B7': (800, 590),                                       'B10': (1240, 590),
    'C1': (80, 390), 'C2': (260, 390), 'C3': (300, 390), 'C4': (500, 390), 'C5': (540, 390), 'C6': (760, 390), 'C7': (800, 390), 'C8': (1070, 390),                    'C10': (1240, 390),
    'D1': (80, 350), 'D2': (260, 350), 'D3': (300, 350), 'D4': (500, 350), 'D5': (540, 350), 'D6': (760, 350), 'D7': (800, 350), 'D8': (1070, 350), 'D9': (1110, 350), 'D10': (1240, 350),
                                                         'E4': (500, 150), 'E5': (540, 150), 'E6': (760, 150), 'E7': (800, 150), 'E8': (1070, 150), 'E9': (1110, 150), 'E10': (1240, 150)
}


graph = Graph()

# List the inital edge between all the nodes, when there is no congestion
graph.addEdge('A1', 'A2', 66)
graph.addEdge('A2', 'A3', 20)
graph.addEdge('A3', 'A4', 120)
graph.addEdge('A4', 'A5', 15)
graph.addEdge('A5', 'A6', 200)
graph.addEdge('A6', 'A7', 16)
graph.addEdge('A7', 'A10', 230)

graph.addEdge('B1', 'B2', 56)
graph.addEdge('B2', 'B3', 11)
graph.addEdge('B3', 'B4', 136)
graph.addEdge('B4', 'B5', 13)
graph.addEdge('B5', 'B6', 200)
graph.addEdge('B6', 'B7', 16)
graph.addEdge('B7', 'B10', 230)

graph.addEdge('C1', 'C2', 36)
graph.addEdge('C2', 'C3', 10)
graph.addEdge('C3', 'C4', 160)
graph.addEdge('C4', 'C5', 10)
graph.addEdge('C5', 'C6', 200)
graph.addEdge('C6', 'C7', 16)
graph.addEdge('C7', 'C8', 184)
graph.addEdge('C8', 'C10', 50)

graph.addEdge('D1', 'D2', 36)
graph.addEdge('D2', 'D3', 10)
graph.addEdge('D3', 'D4', 160)
graph.addEdge('D4', 'D5', 16)
graph.addEdge('D5', 'D6', 200)
graph.addEdge('D6', 'D7', 16)
graph.addEdge('D7', 'D8', 184)
graph.addEdge('D8', 'D9', 7)
graph.addEdge('D9', 'D10', 40)

graph.addEdge('E4', 'E5', 16)
graph.addEdge('E5', 'E6', 200)
graph.addEdge('E6', 'E7', 16)
graph.addEdge('E7', 'E8', 184)
graph.addEdge('E8', 'E9', 7)
graph.addEdge('E9', 'E10', 40)

graph.addEdge('A1', 'B1', 14)
graph.addEdge('A2', 'B2', 16)
graph.addEdge('A3', 'B3', 16)
graph.addEdge('A4', 'B4', 15)
graph.addEdge('A5', 'B5', 15)
graph.addEdge('A6', 'B6', 15)
graph.addEdge('A7', 'B7', 15)
graph.addEdge('A10', 'B10', 15)

graph.addEdge('B1', 'C1', 56)
graph.addEdge('B2', 'C2', 60)
graph.addEdge('B3', 'C3', 60)
graph.addEdge('B4', 'C4', 56)
graph.addEdge('B5', 'C5', 56)
graph.addEdge('B6', 'C6', 56)
graph.addEdge('B7', 'C7', 56)
graph.addEdge('B10', 'C10', 56)

graph.addEdge('C1', 'D1', 10)
graph.addEdge('C2', 'D2', 10)
graph.addEdge('C3', 'D3', 10)
graph.addEdge('C4', 'D4', 10)
graph.addEdge('C5', 'D5', 10)
graph.addEdge('C6', 'D6', 10)
graph.addEdge('C7', 'D7', 10)
graph.addEdge('C8', 'D8', 8)
graph.addEdge('C10', 'D10', 12)

graph.addEdge('D4', 'E4', 56)
graph.addEdge('D5', 'E5', 56)
graph.addEdge('D6', 'E6', 56)
graph.addEdge('D7', 'E7', 56)

graph.addEdge('D10', 'E10', 56)

#Refresh the pygame at a interval of 200ms to check if there arent any modification in the traffic
updateInterval = 0.2  
simulationTime = 30  
edgeUpdates = csvFileEdges('edgeUpdates.csv')

updateTime = time.time()
startSimulation = updateTime
timestamp = 0

# Initialize path and totalDist variables outside the loop
path = []
totalDist = None


#running the simulation
running = True
while running:
    currentTime = time.time()
    elapsedTime = currentTime - startSimulation
    if elapsedTime >= simulationTime:
        running = False

    # Calculate the simulation time
    simulation_minutes = (elapsedTime / simulationTime) * 1440
    hours, minutes = divmod(simulation_minutes, 60)
    time_display = f"{int(hours):02d}:{int(minutes):02d}"

    if currentTime - updateTime >= updateInterval:
        if timestamp < len(edgeUpdates):
            current_updates = edgeUpdates[edgeUpdates['time_step'] == timestamp]
            update_specific_edges(graph, current_updates)
            path, totalDist = dijkstraAlgo(graph, 'A1', 'E10')  # Define the starting node and destination 
            updateTime = currentTime
            timestamp += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #Display the weights of the edges (time) on the display
    screen.fill(backgroundColor)
    for fromNode, edges in graph.nodes.items():
        from_pos = nodePositions[fromNode]
        for toNode, weight in edges.items():
            to_pos = nodePositions[toNode]
            pygame.draw.line(screen, edgeColor, from_pos, to_pos, 1)
            middle_x = (from_pos[0] + to_pos[0]) // 2
            middle_y = (from_pos[1] + to_pos[1]) // 2
            weight_text = font.render(str(weight), True, textColor)
            screen.blit(weight_text, (middle_x, middle_y))
    for node, pos in nodePositions.items():
        pygame.draw.circle(screen, nodeColor, pos, 8)

    if path:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, pathColor, nodePositions[path[i]], nodePositions[path[i + 1]], 4)

    # Display shortest path time 
    if totalDist is not None:
        mins, secs = divmod(int(totalDist), 60)
        timeDisplay = font.render(f"Shortest Path Time: {mins:02d}:{secs:02d} seconds", True, textColor)
        screen.blit(timeDisplay, (10, 30))  # Position the text below the simulation time

    pygame.display.flip()

pygame.quit()
