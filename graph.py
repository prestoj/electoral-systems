class Graph(object):

    def __init__(self, keys):
        self.n_nodes = len(keys)
        self.edge_matrix = {}
        for key in keys:
            edges = {}
            for key_ in keys:
                edges[key_] = 0
            self.edge_matrix[key] = edges

    def add_edge(self, i, j):
        self.edge_matrix[i][j] = 1

    def remove_edge(self, i, j):
        self.edge_matrix[i][j] = 0

    def has_cycle(self, i):
        visited = {key: False for key in self.edge_matrix}

        return self._explore_children(i, i, visited)

    def get_sources(self):
        sources = []
        for i in self.edge_matrix.keys():
            if self._is_source(i):
                sources.append(i)

        # if len(sources) == 1:
        return sources
        #
        # sources_sum = []
        # for source in sources:
        #     sum = 0
        #     for value in self.edge_matrix[source].values():
        #         sum += value
        #     sources_sum.append(sum)
        #
        # # print(sources_sum, sources)
        # sources_sum, sources = zip(*sorted(zip(sources_sum, sources)))
        # print(list(sources_sum), list(sources))
        #
        # return list(sources)

    def _is_source(self, a):
        for i in self.edge_matrix.keys():
            if self.edge_matrix[i][a] != 0:
                return False

        return True


    def _explore_children(self, a, b, visited):
        for i in self.edge_matrix.keys():
            if (self.edge_matrix[a][i] != 0 and not visited[i]):
                if self._can_reach(i, b, visited):
                    return True
                else:
                    visited[i] = True

        return False

    def _can_reach(self, a, b, visited):
        if a == b:
            return True

        visited[a] = True

        return self._explore_children(a, b, visited)
