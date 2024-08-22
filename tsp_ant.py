import random

# δ (u, v) -> distance between 2 edges    (distance)
# τ (u, v) -> desirability of edge        (pheromone)

# η -> heuristic of some means ( 1 / δ )  (weight)

def get(mean, std):
    x = random.gauss(mu=mean, sigma=std)
    if x > 0:
        return x
    return get(mean, std)


class TSPAnt:
    def __init__(self, id, source, model, init_pheromone, q_0=0.9, alpha=1.0, beta=5.0) -> None:
        self.id = id
        self.source = source
        self.visited = [source]

        self.current = source
        self.distance = 0
        self.model = model
        self.alpha = get(1, 1)
        self.beta = get(5, 1)
        self.init_pheromone = init_pheromone
        self.q_0 = q_0

    def exploitation(self, neighbors):
        # j = argmax{ [τ(i,l)]^α * [η(i,l)]^β } (exploitation)
        best_value = float("-inf")
        next_city = None

        for neighbor in neighbors:
            tau_ij = self.model.graph[self.current][neighbor]["pheromone"]
            n_ij = self.model.graph[self.current][neighbor]["weight"]

            this_value = (tau_ij**self.alpha) * (n_ij**self.beta)
            if best_value < this_value:
                best_value = this_value
                next_city = neighbor

        return next_city

    def exploration(self, neighbors):
        # p(i,j) = [τ(i,j)]^α * [η(i,j)]^β / Σ [τ(i,l)]^α * [η(i,l)]^β
        sum_of_values = 0
        individual_values = []

        for neighbor in neighbors:
            tau_ij = self.model.graph[self.current][neighbor]["pheromone"]
            n_ij = self.model.graph[self.current][neighbor]["weight"]
            this_value = (tau_ij**self.alpha) * (n_ij**self.beta)

            sum_of_values += this_value
            individual_values.append(this_value)

        probabilities = [value / sum_of_values for value in individual_values]
        next_city = random.choices(neighbors, weights=probabilities)[0]

        return next_city

    def update_local_pheromone(self, next_city, rho=0.1):
        # [τ(i,j)] = (1 - ρ) * (τ(i,j)) +  ρ * [Δ τ(i,j)]
        pheromone_current_next = self.model.graph[self.current][next_city]["pheromone"]
        new_pheromone = (1 - rho) * pheromone_current_next + rho * self.init_pheromone
        self.model.graph[self.current][next_city]["pheromone"] = new_pheromone

    def next_city(self):
        neighbors = list(self.model.graph.neighbors(self.current))
        neighbors = [i for i in neighbors if i not in self.visited]
        if not neighbors:
            return self.source

        q = random.uniform(0, 1)
        if q < self.q_0:
            next_city = self.exploitation(neighbors)
        else:
            next_city = self.exploration(neighbors)

        return next_city

    def step(self):
        new_city = self.next_city()
        self.distance += self.model.graph[self.current][new_city]["distance"]
        self.visited.append(new_city)
        # print(f"Ant {self.id} moved from {self.current} to {new_city}")
        self.update_local_pheromone(new_city)
        self.current = new_city

    def next_city_greedy(self):
        neighbors = list(self.model.graph.neighbors(self.current))
        best_distance = float("inf")
        new_city = None

        for nei in neighbors:
            if nei in self.visited:
                continue

            distance = self.model.graph[self.current][nei]["distance"]
            if distance < best_distance:
                best_distance = distance
                new_city = nei

        if new_city is None:
            new_city = self.source

        return new_city
