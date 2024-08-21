import networkx as nx
from tsp_ant import TSPAnt
import matplotlib.pyplot as plt
import random
import time

# δ (u, v) -> distance between 2 edges    (distance)
# τ (u, v) -> desirability of edge        (pheromone)

# η -> heuristic of some means ( 1 / δ )  (weight)


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def generate_color(weight):
    weight = weight / 1.414
    green = int((1 - weight) * 255)
    red = int(weight * 255)
    blue = 0
    hex = f"#{red:02X}{green:02X}{blue:02X}"
    return hex


class TSPGraph:
    def __init__(self, n, m, initial_pheromone=10e-6):
        self.n_nodes = n
        self.n_ants = m
        self.graph = nx.random_geometric_graph(n=self.n_nodes, radius=2).to_directed()
        self.init_pheromone = initial_pheromone

        # graph essentials
        self.positions = self.get_positions()
        self.cities = self.get_cities()
        self.ants = self.place_ants()

        # visual
        self.colors = self.place_weight_and_get_color()

        # performance
        self.running = True
        self.steps = 0
        self.ants_distances = [[0] for _ in range(self.n_ants)]
        self.initital_best_value = float("inf")
        self.best_route = None
        self.best_distance = float("inf")

        self.best_record = []
        self.progress_record = []

    def reset(self):
        self.ants = self.place_ants()  # Different ant objects

        self.running = True
        self.steps = 0
        self.ants_distances = [[0] for _ in range(self.n_ants)]
        # self.best_route = None
        # self.best_distance = float("inf")

    def get_positions(self):
        positions = {}
        for k, v in dict(self.graph.nodes.data()).items():
            positions[k] = v["pos"]
        return positions

    def parse_file(self, nodes, positions):
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
        self.ants_distances = [[0] for _ in range(self.n_ants)]
        self.best_route = None
        self.best_distance = float("inf")

    def get_cities(self):
        cities = set()
        for i in range(self.n_nodes):
            cities.add(i)
        return cities

    def place_ants(self):
        selected = set()
        ants = []
        for i in range(self.n_ants):
            city = random.choice(list(self.cities - selected))
            ant = TSPAnt(i, city, self, self.init_pheromone)
            ants.append(ant)
            # print(f"Ant {i} placed in city {city}")
        return ants

    def place_weight_and_get_color(self):
        colors = []
        for u, v in self.graph.edges:
            pos_u = self.positions[u]
            pos_v = self.positions[v]
            self.graph[u][v]["distance"] = euclidean_distance(pos_u, pos_v)
            self.graph[u][v]["weight"] = 1 / self.graph[u][v]["distance"]
            self.graph[u][v]["pheromone"] = self.init_pheromone
            colors.append(generate_color(self.graph[u][v]["distance"]))
        return colors

    def draw_graph(self):
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
        )

        fig.savefig('graph.png')
        plt.close(fig)
        return fig
    
    def shortest_plot(self):
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
        fig.savefig('shortest_plot.png')
        plt.close(fig)
        return fig

    def run(self):
        while self.running:
            self.step()

    def update_global_pheromone(self, rho=0.1):
        delta_tau = {edge: 0 for edge in self.graph.edges}

        for i in range(len(self.best_route) - 1):
            u, v = self.best_route[i], self.best_route[i + 1]
            delta_tau[(u, v)] += 1 / self.best_distance

        for u, v in self.graph.edges:
            pheromone_uv = self.graph[u][v]["pheromone"]
            new_tau_ij = (1 - rho) * pheromone_uv + rho * delta_tau.get((u, v), 0)
            self.graph[u][v]["pheromone"] = new_tau_ij

    def step(self):
        if not self.running:
            return

        for ant in self.ants:
            ant.step()
            self.ants_distances[ant.id].append(ant.distance)

        self.steps += 1
        if self.steps >= self.n_nodes:  # Reached source again
            current_best = float("inf")
            for ant in self.ants:
                if ant.distance < current_best:
                    current_best = ant.distance
                if ant.distance < self.best_distance:
                    if self.initital_best_value == float("inf"):
                        self.initital_best_value = ant.distance
                    self.best_distance = ant.distance
                    self.best_route = ant.visited

            self.best_record.append(self.best_distance)
            self.progress_record.append(current_best)
            self.update_global_pheromone()
            self.running = False

    def progress_plot(self):
        plt.clf()
        steps = range(0, len(self.progress_record))
        fig, ax = plt.subplots(figsize=(4, 4))

        # print(steps, self.progress_record)
        ax.plot(steps, self.progress_record, linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Shortest distance')
        ax.set_xlim(0)
        ax.set_ylim(self.best_distance - (self.best_distance * 5 / 100) if self.best_distance != float("inf") else 1, 
                    self.initital_best_value + (self.initital_best_value * 5 / 100) if self.initital_best_value != float("inf") else 1)
        ax.set_title('Progress till now')
        fig.savefig('progress_plot.png')

        plt.close(fig)
        return fig
    
    def best_plot(self):
        plt.clf()
        steps = range(0, len(self.best_record))
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(steps, self.best_record, linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Shortest distance')
        ax.set_xlim(0)
        ax.set_ylim(self.best_distance - (self.best_distance * 5 / 100) if self.best_distance != float("inf") else 1, 
                    self.initital_best_value + (self.initital_best_value * 5 / 100) if self.initital_best_value != float("inf") else 1)
        ax.set_title('Best till now')
        plt.close(fig)
        fig.savefig('best_plot.png')
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
