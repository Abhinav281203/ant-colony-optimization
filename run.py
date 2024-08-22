from tsp_graph import TSPGraph

try:
    with open("./tsp/eil51.tsp", "r") as f:
        lines = f.readlines()

    nodes = 0
    positions = {}

    for i in lines[6:-1]:
        id, x, y = list(map(float, i.split()))
        positions[int(id - 1)] = [x, y]
        nodes += 1

    g = TSPGraph(2, 2)              # generate random
    g.parse_file(nodes, positions)  # parse actual

    best_distance = float("inf")
    best_route = []
    for i in range(5000):
        g.run()
        if g.best_distance < best_distance:
            best_distance = g.best_distance
            best_route = g.best_route

            print(f"New best found at episode {i}: {best_distance}")
        print(f"episode {i}: {g.best_distance}")
        g.reset()

    print(f"Final: {best_distance}")
    print(best_route)
    g.shortest_plot()
    g.progress_plot()
    g.best_plot()
    g.draw_graph()

except KeyboardInterrupt:
    print("\nAborted")
