from tsp_graph import TSPGraph


def main():
    try:
        with open("./tsp/eil51.tsp", "r") as f:
            lines = f.readlines()

        nodes = 0
        positions = {}

        for i in lines[6:-1]:
            id, x, y = list(map(float, i.split()))
            positions[int(id - 1)] = [x, y]
            nodes += 1

        g = TSPGraph(2, 2)  # generate random
        g.parse_file(nodes, positions)  # parse actual

        best_distance = float("inf")
        best_route = []
        for i in range(5000):
            g.run()
            if g.best_distance < best_distance:
                best_distance = g.best_distance
                best_route = g.best_route

                print(f"New best found at Run {i}: {best_distance}")
            print(f"Run {i}: {g.best_distance}")
            g.reset()

        print(f"Final: {best_distance}")
        print(best_route)
        g.draw_graph()
        g.draw_path()
        g.progress_plot()
        g.best_plot()

    except KeyboardInterrupt:
        print("\nAborted")


if __name__ == "__main__":
    main()
