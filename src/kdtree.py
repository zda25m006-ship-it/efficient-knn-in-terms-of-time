class KDNode:
    def __init__(self, point, label, axis, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, points, labels):
        self.k = len(points[0])
        self.root = self._build(points, labels, depth=0)

    def _build(self, points, labels, depth):
        if not points:
            return None

        axis = depth % self.k
        data = list(zip(points, labels))
        data.sort(key=lambda x: x[0][axis])

        median = len(data) // 2

        return KDNode(
            point=data[median][0],
            label=data[median][1],
            axis=axis,
            left=self._build(
                [x[0] for x in data[:median]],
                [x[1] for x in data[:median]],
                depth + 1
            ),
            right=self._build(
                [x[0] for x in data[median + 1:]],
                [x[1] for x in data[median + 1:]],
                depth + 1
            )
        )
