from algorithms.classic.a_star import AStar
from structures import Point


class Dijkstra(AStar):
    def f(self, x: Point) -> float:
        return self.mem.g[x]
