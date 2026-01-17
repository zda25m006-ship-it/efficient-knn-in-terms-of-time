import heapq
from collections import Counter
from src.kdtree import KDTree


class KNNClassifier:
    def __init__(self, k=5, distance_fn=None, weighted=False):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.tree = None

    def fit(self, X, y):
        self.tree = KDTree(X, y)

    def _search(self, node, target, heap):
        if node is None:
            return

        dist = self.distance_fn(target, node.point)
        heapq.heappush(heap, (-dist, node.label))
        if len(heap) > self.k:
            heapq.heappop(heap)

        axis = node.axis
        diff = target[axis] - node.point[axis]

        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._search(near, target, heap)

        if abs(diff) < -heap[0][0]:
            self._search(far, target, heap)

    def predict_one(self, x):
        heap = []
        self._search(self.tree.root, x, heap)

        if self.weighted:
            votes = {}
            for dist, label in heap:
                weight = 1 / (abs(dist) + 1e-9)
                votes[label] = votes.get(label, 0) + weight
            return max(votes, key=votes.get)
        else:
            labels = [label for _, label in heap]
            return Counter(labels).most_common(1)[0][0]

    def predict(self, X):
        return [self.predict_one(x) for x in X]
