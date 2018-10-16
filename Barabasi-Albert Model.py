
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import powerlaw

# Computes the degree distribution of an input graph
def Distribution(graph): 
    DegreeList = sorted([d for n, d in graph.degree()])  

    degreeCount = [0]*(max(DegreeList)+1)
    for n in DegreeList:
        degreeCount[n] = degreeCount[n] + 1 

    degreeDistribution = [n * (1/float(len(DegreeList))) for n in degreeCount]
    return degreeDistribution


G = nx.complete_graph(4)

# Implements the Barabasi-Albert model. 
# Returns the degree distribution of the graph at select points and tracks degree dynamics. 
# DistributionPoints is a list of time steps at which to compute the degree distribution.
def BA_Model(N, m, InitialGraph, DistributionPoints):
    G = InitialGraph
    # Create list where the nth entry holds the degree of the nth node
    DegreeList = [3,3,3,3]
    # Initialize lists for tracking degree dynamics, degree distribution, and average clustering coefficient later
    Degrees0 = [3]
    Degrees100 = []
    Degrees1000 = []
    DegreeDistributions = []
    AvgClustering = []    
    
    # Add new nodes and preferentially attach to existing nodes    
    for n in range(m,N):
        # Add new node to graph
        G.add_node(n)
        # Randomly select nodes for new node to attach to
        NewEdgeTargets = []
        NewEdgeTargets = np.random.choice(a = range(n), size = m, replace = False, p = [i/float(sum(DegreeList)) for i in DegreeList]) 
        # Add to list to reflect that the newly-added node will have m connections
        DegreeList.append(m)
        
        # Update lists to reflect new edges
        for j in NewEdgeTargets:
            DegreeList[j] = DegreeList[j] + 1
            G.add_edge(n,j)
        
        # Compute and track average clustering coefficient
        AvgClustering.append(nx.average_clustering(G))
        
        # Compute and report the degree distribution at the desired time steps
        if n in DistributionPoints:
            DegreeDistributions.append(Distribution(G))
                            
        # Track degree of selected nodes over time
        Degrees0.append(DegreeList[0])
        if n >= 100: 
            Degrees100.append(DegreeList[100])
        if n >= 1000: 
            Degrees1000.append(DegreeList[1000])    
    
    return DegreeDistributions, AvgClustering, Degrees0, Degrees100, Degrees1000

# Run the algorithm and record the results
Results = BA_Model(2500,4,nx.complete_graph(4),[100,1000,2450])
         
# Define x and y variables for plotting degree distributions 
x1 = [i for i in range(len(Results[0][0]))]
y1 = Results[0][0]
x2 = [i for i in range(len(Results[0][1]))]
y2 = Results[0][1]
x3 = [i for i in range(len(Results[0][2]))]
y3 = Results[0][2]  


# Plot degree distribution and line of best fit with log-log scale
from matplotlib import style
style.use('ggplot')
plt.subplot(1,2,1)
plt.loglog(x1, y1, 'ro', x2, y2, 'bo', x3, y3, 'go')
plt.title("Degree Distribution")
plt.xlabel("k")
plt.ylabel("p(k)")
plt.show() 


# Plot cumulative degree distribution
CumDegree1 = np.cumsum(y1)
CumDegree2 = np.cumsum(y2)
CumDegree3 = np.cumsum(y3)
plt.subplot(1,2,2)
plt.plot(x1,CumDegree1, 'ro', x2, CumDegree2, 'bo', x3, CumDegree3, 'go')
plt.title("Cumulative Degree Distribution")
plt.xlabel("k")
plt.ylabel("Cumulative Frequency")
plt.show() 

   
# Plot average clustering coefficient over time
plt.figure()
plt.plot([i for i in range(len(Results[1]))], Results[1])
plt.title("Average Clustering Coefficient Over Time")
plt.xlabel("t")
plt.ylabel("Average Clustering Coefficient")
plt.show()


# Plot degree dynamics over time
plt.figure()
x_1 = [i for i in range(len(Results[2]))]
x_2 = [i for i in range(len(Results[3]))]
x_3 = [i for i in range(len(Results[4]))]
plt.plot(x_1, Results[2], 'ro', x_2, Results[3], 'bo', x_3, Results[4], 'go')
plt.title("Degree of Selected Nodes Over Time")
plt.xlabel("t")
plt.ylabel("k")
plt.show()