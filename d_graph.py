# Course: CS261 - Data Structures
# Author:   Justin Canzonetta
# Assignment:   6 Part 2
# Description:  Implementation of a DirectedGraph class.

from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Adds a new vertex to the graph.
        """
        self.v_count += 1

        # Update the existing vertices in adj_matrix with an additional adjancency vertex.
        for vertex in self.adj_matrix:
            vertex.append(0)

        # Add the new vertex to the end with initial values 0.
        self.adj_matrix.append([0]*self.v_count)

        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Takes two integers and adds an edge from the first vertex to the second. If the third optional
        parameter is not provided, the weight is set as 1.

        Both vertices must be already a part of the graph and loops are not allowed.
        """
        # Check that the input is valid.
        if src >= self.v_count or src < 0 or dst >= self.v_count or dst < 0 or src == dst or weight < 1:
            return

        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Takes two integers and removes the edge from the first vertex to the second.
        """
        # Check that the input is valid.
        if src >= self.v_count or src < 0 or dst >= self.v_count or dst < 0 or src == dst:
            return

        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        Returns an array containing all of the vertices in the graph.
        """
        output_arr = list()
        for i in range(self.v_count):
            output_arr.append(i)

        return output_arr

    def get_edges(self) -> []:
        """
        Returns an array containing all of the edges in the graph.

        Each edge is provided as a tuple of two incident vertex indices and weight.
        """
        output_arr = list()
        # src is the vertex indicie of the source vertex in each iteration.
        for src, src_arr in enumerate(self.adj_matrix):
            # dst is the vertex indicie of the destination vertex in each iteration.
            for dst, dst_weight in enumerate(src_arr):
                if dst_weight != 0:
                    output_arr.append((src, dst, dst_weight))

        return output_arr

    def is_valid_path(self, path: []) -> bool:
        """
        Takes an array of vertex indices and returns True if the path is valid and False otherwise.
        """
        # If the array provided is an empty array, the path is valid.
        if not path:
            return True

        # Check if the first vertex in the path is valid.
        if path[0] < len(self.adj_matrix) and path[0] >= 0:
            prev_vertex = path[0]
        else:
            return False

        # Check if the next vertex in the path is connected to the previous vertex analyzed.
        for i in range(1, len(path)):
            # If the next vertex is not a valid vertex, or there is not an edge connecting it to the
            # previous vertex, return False. Otherwise continue to the next vertex in the path.
            if path[i] >= len(self.adj_matrix) or path[i] < 0 or self.adj_matrix[prev_vertex][path[i]] <= 0:
                return False

            prev_vertex = path[i]

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Takes an initial vertex index and an optional ending vertex index and returns an array of verticies
        visited during a depth first search starting at the initial vertex.

        When multiple vertices are incident to the currently visited index, the next index visited will be the
        in ascending order.
        """

        # End the function early if v_start is not a vertex in the graph.
        if v_start < 0 or v_start >= len(self.adj_matrix):
            return []
        else:
            visited_vertices = list()
            vertex = v_start
            stack = deque()
            stack.append(vertex)

        while len(stack) > 0 and vertex != v_end:
            # The next vertex analyzed is always the last vertex added to the stack.
            vertex = stack.pop()

            if vertex not in visited_vertices:
                visited_vertices.append(vertex)

                # Sort all of the adjacent verticies of the current vertex in reverse order
                # since they will be added at the top of the stack from first to last.
                adjacent_verticies = list()
                for i, weight in enumerate(self.adj_matrix[vertex]):
                    if weight > 0:
                        adjacent_verticies.append(i)
                adjacent_verticies.sort(reverse=True)

                for neighbor in adjacent_verticies:
                    stack.append(neighbor)

        return visited_vertices

    def bfs(self, v_start, v_end=None) -> []:
        """
        TODO: Write this implementation
        """
        pass

    def has_cycle(self):
        """
        TODO: Write this implementation
        """
        pass

    def dijkstra(self, src: int) -> []:
        """
        TODO: Write this implementation
        """
        pass


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)

    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)

    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
