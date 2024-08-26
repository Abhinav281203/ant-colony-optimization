import networkx as nx
from tsp_ant import TSPAnt
import matplotlib.pyplot as plt
import random

# δ (u, v) -> distance between 2 edges    (distance)
# τ (u, v) -> desirability of edge        (pheromone)
# η -> heuristic of some means ( 1 / δ )  (weight)
# The shorter the distance higher the heuristic measure


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

class TSPGraph:
    def __init__(self, n, m, initial_pheromone=10e-6):
        self.n_nodes = n
        self.n_ants = m
        self.graph = nx.random_geometric_graph(n=self.n_nodes, radius=2).to_directed()
        self.init_pheromone = initial_pheromone

        self.positions = self.get_positions()
        self.cities = self.get_cities()
        self.ants = self.place_ants()

        self.colors = self.place_weight_and_get_color()

        self.running = True
        self.steps = 0
        self.initital_best_value = float("inf")
        self.best_route = None
        self.best_distance = float("inf")
        self.record_best = []
        self.record_progress = []

    def reset(self):
        # Reset doesn't involve changing distances measured so far
        # Reset ants and set steps to 0
        for ant in self.ants:
            ant.reset()
        self.running = True
        self.steps = 0

    def get_positions(self):
        # Get the positions of the nodes for calculating distances or drawing graph
        positions = {}
        for k, v in dict(self.graph.nodes.data()).items():
            positions[k] = v["pos"]
        return positions
    
    def parse_file(self, nodes, positions):
        # Using a tsp file to build the graph, all attributes needs to be set manually
        self.graph = nx.Graph().to_directed()
        for i in range(nodes):
            x, y = positions[i]
            self.graph.add_node(i, pos=(x, y))

        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    continue
                self.graph.add_edge(i, j)

        self.n_nodes = nodes
        self.n_ants = nodes
        self.positions = self.get_positions()
        self.cities = self.get_cities()
        self.ants = self.place_ants()
        self.colors = self.place_weight_and_get_color()
        self.running = True
        self.steps = 0
        self.best_route = None
        self.best_distance = float("inf")

    def get_cities(self):
        cities = set()
        for i in range(self.n_nodes):
            cities.add(i)
        return cities
    
    def place_ants(self):
        # Creates the ants and places them on a random empty city
        selected = set()
        ants = []
        for i in range(self.n_ants):
            city = random.choice(list(self.cities - selected))
            ant = TSPAnt(i, city, self, self.init_pheromone)
            ants.append(ant)
        return ants
    
    def place_weight_and_get_color(self):
        # Places the distance, weight and initial pheromone on each edge
        colors = []
        for u, v in self.graph.edges:
            pos_u = self.positions[u]
            pos_v = self.positions[v]
            self.graph[u][v]["distance"] = euclidean_distance(pos_u, pos_v)
            self.graph[u][v]["weight"] = 2 / self.graph[u][v]["distance"]
            self.graph[u][v]["pheromone"] = self.init_pheromone
            # colors.append(generate_color(self.graph[u][v]["distance"]))
        return colors
    
    def update_global_pheromone(self, rho=0.1):
        # Updates the global pheromone on the shortest path found so far
        # Evaporation is done on all the edges
        # τ(i, j) = (1 - α) * τ(i, j) + α * Δτ(i, j)
        # Where, Δτ(i, j) = 1/Lgb if edge (i, j) belongs to the global best tour, 0 otherwise
        # Lgb = length of global best distance
        delta_tau = {edge: 0 for edge in self.graph.edges}

        for i in range(len(self.best_route) - 1):
            u, v = self.best_route[i], self.best_route[i + 1]
            delta_tau[(u, v)] += 1 / self.best_distance

        for u, v in self.graph.edges:
            pheromone_uv = self.graph[u][v]["pheromone"]
            new_tau_ij = (1 - rho) * pheromone_uv + rho * delta_tau.get((u, v), 0)
            self.graph[u][v]["pheromone"] = new_tau_ij

    def collect_data(self):
        this_run_best = float("inf")
        for ant in self.ants:
            if ant.distance < this_run_best:
                this_run_best = ant.distance

            if ant.distance < self.best_distance:
                if self.initital_best_value == float("inf"):
                    self.initital_best_value = ant.distance

                self.best_distance = ant.distance
                self.best_route = ant.visited

        return this_run_best

    def step(self):
        # One step makes all the ants make a step (going to other cities)
        if not self.running:
            return

        for ant in self.ants:
            ant.step()

        self.steps += 1
        if self.steps >= self.n_nodes:  # Reached source again (one run)
            this_run_best = self.collect_data()

            self.record_best.append(self.best_distance)
            self.record_progress.append(this_run_best)
            self.update_global_pheromone()
            self.running = False

    def run(self):
        # Executes the step until all the ants visits all cities and comes back to source
        while self.running:
            self.step()

    def draw_graph(self, name="full_graph"):
        # Draws the complete graph, colors are only used for a graph under unit square
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4))

        nx.draw(
            self.graph,
            pos=self.positions,
            # edge_color=self.colors,
            node_size=50,
            arrowstyle="-",
            with_labels=False,
            ax=ax,
            width=0.3
        )

        fig.savefig(f"{name}.png")
        plt.close(fig)
        return fig
    
    def draw_path(self, name="path_graph"):
        # Creates a temp graph with the shortest path found by the ants and creates a figure
        if not self.best_route:
            return None
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4))
        temp_graph = nx.Graph().to_directed()
        temp_pos = {}
        for ele in self.best_route:
            temp_pos[ele] = self.positions[ele]
            temp_graph.add_node(ele)
        
        for i in range(len(self.best_route) - 2):
            temp_graph.add_edge(self.best_route[i], self.best_route[i + 1])
        
        nx.draw(
            temp_graph,
            pos=temp_pos,
            node_size=50,
            arrowstyle="-",
            with_labels=False,
            ax=ax,
        )
        fig.savefig(f"{name}.png")
        plt.close(fig)
        return fig
    
    def progress_plot(self, name="progress_plot"):
        # Plots the best distance of each run
        plt.clf()
        steps = range(0, len(self.record_progress))
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(steps, self.record_progress, linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Shortest distance')
        ax.set_xlim(0)
        ax.set_ylim(self.best_distance - (self.best_distance * 5 / 100) if self.best_distance != float("inf") else 1, 
                    self.initital_best_value + (self.initital_best_value * 5 / 100) if self.initital_best_value != float("inf") else 1)
        ax.set_title('Progress till now')
        fig.savefig(f"{name}.png")

        plt.close(fig)
        return fig
    
    def best_plot(self, name="best_plot"):
        # Plots the best distance discovered among all runs so far
        plt.clf()
        steps = range(0, len(self.record_best))
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(steps, self.record_best, linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Shortest distance')
        ax.set_xlim(0)
        ax.set_ylim(self.best_distance - (self.best_distance * 5 / 100) if self.best_distance != float("inf") else 1, 
                    self.initital_best_value + (self.initital_best_value * 5 / 100) if self.initital_best_value != float("inf") else 1)
        ax.set_title('Best till now')
        fig.savefig(f"{name}.png")
        plt.close(fig)
        return fig
    
    
if __name__ == "__main__":
    g = TSPGraph(20, 15)
    best_distance = float("inf")
    best_route = []
    for i in range(2000):
        g.run()
        if g.best_distance < best_distance:
            best_distance = g.best_distance
            best_route = g.best_route

            print(f"New best found at episode {i}: {best_distance}")
        print(f"episode {i}: {g.best_distance}")
        g.reset()

    print(f"Final: {best_distance}")