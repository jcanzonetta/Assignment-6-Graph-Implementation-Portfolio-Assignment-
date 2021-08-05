# Course:   CS261 Data Structures
# Author:   Justin Canzonetta
# Assignment:   6 Part 1
# Description:  Implementation of an UndirectedGraph class.

from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph
        """

        if v not in self.adj_list:
            self.adj_list[v] = []

    def add_edge(self, u: str, v: str) -> None:
        """
        Add edge to the graph
        """

        # Per requirement, only add edges between two different verticies.
        if u == v:
            return

        # Add verticies u and v if not already precent in the adj_list.
        if u not in self.adj_list:
            self.add_vertex(u)
        if v not in self.adj_list:
            self.add_vertex(v)

        # Add each vertex to the opposing vertex's list of edges.
        if u not in self.adj_list[v] and v not in self.adj_list[u]:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Remove edge from the graph
        """

        # Only remove an edge between two verticies if both verticies exist in the adj_list.
        if u in self.adj_list and v in self.adj_list:
            # Only remove an edge if it exists in the adj_list for both verticies.
            if v in self.adj_list[u] and u in self.adj_list[v]:
                self.adj_list[u].remove(v)
                self.adj_list[v].remove(u)

    def remove_vertex(self, v: str) -> None:
        """
        Remove vertex and all connected edges
        """

        # Only remove the vertex if it exists in the adj_list.
        if v in self.adj_list:
            # Remove any edge that includes v in all of the neighbors of v.
            for neighbor in self.adj_list[v]:
                self.adj_list[neighbor].remove(v)

            # Remove the vertex v itself.
            del self.adj_list[v]

    def get_vertices(self) -> []:
        """
        Return list of vertices in the graph (any order)
        """

        output_arr = list()
        for vertex in self.adj_list:
            output_arr.append(vertex)

        return output_arr

    def get_edges(self) -> []:
        """
        Return list of edges in the graph (any order)
        """

        output_arr = list()

        for vertex in self.adj_list:
            for neighbor in self.adj_list[vertex]:
                # Create a tuple in alphabetical order to check if already present.
                if vertex < neighbor:
                    temp_tuple = (vertex, neighbor)
                else:
                    temp_tuple = (neighbor, vertex)

                # Check if the tuple is in the output array and add it if not.
                if temp_tuple not in output_arr:
                    output_arr.append(temp_tuple)

        return output_arr

    def is_valid_path(self, path: []) -> bool:
        """
        Return true if provided path is valid, False otherwise
        """

        # A path with no verticies is always a valid path.
        if len(path) == 0:
            return True

        # Check if the first vertex in the path is present in the adj_list.
        # Initialize the prev_vertex to it if it exists for iterative comparison.
        if path[0] in self.adj_list:
            prev_vertex = path[0]
        else:
            return False

        # Check if the next vertex in the path is in the list of neighbors of the
        # prev_vertex, iterating to the last element in path.
        for i in range(1, len(path)):
            if path[i] not in self.adj_list[prev_vertex]:
                # If the neighbor is ever not found, return False.
                return False
            prev_vertex = path[i]

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search
        Vertices are picked in alphabetical order
        """

        # End the function early if v_start is not a vertex in the graph.
        if v_start not in self.adj_list:
            return []
        else:
            visited_verticies = list()
            vertex = v_start
            stack = deque(vertex)  # Used as a stack to scan depth first.

        while len(stack) > 0 and vertex != v_end:
            # The next vertex analyszed is always the last vertex added to the stack.
            vertex = stack.pop()

            if vertex not in visited_verticies:
                visited_verticies.append(vertex)

                # Sort all of the adjacent verticies of the current vertex in reverse
                # order since they will be added at the top of the stack from first to
                # last.
                sorted_neighbors = sorted(self.adj_list[vertex], reverse=True)

                for neighbor in sorted_neighbors:
                    stack.append(neighbor)

        # Edge case check when v_start == v_end.
        if vertex == v_end and vertex not in visited_verticies:
            visited_verticies.append(vertex)

        return visited_verticies

    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search
        Vertices are picked in alphabetical order
        """
        # End the function early if v_start is not a vertex in the graph.
        if v_start not in self.adj_list:
            return []
        else:
            visited_verticies = list()
            vertex = v_start
            queue = deque(vertex)  # Used as a queue to scan breadth first.

        while len(queue) > 0 and vertex != v_end:
            # The next vertex analyzed is always the oldest vertex in the queue.
            vertex = queue.pop()

            # Only add the vertex from the queue to the list of visited verticies if not already visited.
            if vertex not in visited_verticies:
                visited_verticies.append(vertex)

            # Sort the list of adjacent verticies since they will be added to the front of the queue
            # and removed from the back of the queue.
            sorted_neighbors = sorted(self.adj_list[vertex])

            for neighbor in sorted_neighbors:
                if neighbor not in visited_verticies:
                    # Only add the neighbor to the queue if it's not already visited. If it is already
                    # visited, then the depth at this point is greater and is ignored.
                    queue.appendleft(neighbor)

        # Edge case check when v_start == v_end.
        if vertex == v_end and vertex not in visited_verticies:
            visited_verticies.append(vertex)

        return visited_verticies

    def count_connected_components(self):
        """
        Return number of connected componets in the graph
        """
        # A list of sets of verticies which make up one connected component.
        connected_sets = []
        for vertex in self.adj_list:
            # Perform a depth first search to get a set of connected verticies.
            vertex_set = set(self.dfs(vertex))

            # Check if the set has already been included in the list.
            if vertex_set not in connected_sets:
                connected_sets.append(vertex_set)

        return len(connected_sets)

    def has_cycle(self):
        """
        Return True if graph contains a cycle, False otherwise
        """

        for vertex in self.adj_list:
            # Perform a depth first search to find a connected set of verticies.
            connected_vertex_arr = self.dfs(vertex)

            # Count the total number of edges within that set.
            connected_edge_arr = list()
            for connected_vertex in connected_vertex_arr:
                for connected_neighbor in self.adj_list[connected_vertex]:

                    # Create tuples representing an edge in ascending lexicographical order.
                    if connected_vertex < connected_neighbor:
                        edge_tuple = (connected_vertex, connected_neighbor)
                    else:
                        edge_tuple = (connected_neighbor, connected_vertex)

                    # Check if the edge is in the connected_edge_arr and add it if not.
                    if edge_tuple not in connected_edge_arr:
                        connected_edge_arr.append(edge_tuple)

            # Return True if within this set of connected verticies, the number of edges is
            # greater than or equal to the number of verticies.
            if len(connected_vertex_arr) <= len(connected_edge_arr):
                return True

        return False


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)

    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)

    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
