import math

# Computational Graph
# Calculate the derivative while evaluating the expression
# Use chain rule to get the derivatives by calling derivative(f)
nodes, edges = [], []

class Node:
    # The Number, the index in nodes list, edges
    val, index, in_edge_indices = None, None, []

    # Generate result node and connect computational graph
    def unary_operation(self, result, dfdx):
        edges.append({ "src": self.index, "dval": dfdx })
        return Node(result, [len(edges) - 1])

    def binary_operation(self, other, result, dfdx, dfdy):
        edges.append({ "src": self.index, "dval": dfdx })
        edges.append({ "src": other.index, "dval": dfdy })
        return Node(result, [len(edges) - 2, len(edges) - 1])

    def __init__(self, val, e_indices=[]):
        self.val, self.index, self.in_edge_indices = val, len(nodes), e_indices
        nodes.append(self)

    # Print number
    def __repr__(self):
        return str(self.val)
    
    # Calling unary_operation or binary_operation with result and derivative
    def __add__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val + other, 1)
        return self.binary_operation(other, self.val + other.val, 1, 1)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val - other, 1)
        return self.binary_operation(other, self.val - other.val, 1, -1)

    def __rsub__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val - other, -1)
        return self.binary_operation(other, other.val - self.val, -1, 1)

    def __mul__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val * other, other)
        return self.binary_operation(other, self.val * other.val,
            other.val, self.val)

    def __rmul__(self, other):
        return other.__mul__(self)

    def __truediv__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val/other, 1/other)
        return self.binary_operation(other, self.val/other.val, 1/other.val,
            -self.val / (other.val**2))

    def __rtruediv__(self, other):
        if type(other) != Node:
            return self.unary_operation(other/self.val, -self.val / (other ** 2))
        return self.binary_operation(other, other.val/self.val,
            -self.val / (other ** 2), 1/other.val)

    def __pow__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val**other,
                other * (self.val**(other - 1)))

        dfdx = other.val * (self.val**(other.val - 1))
        dfdy = (self.val**other.val) * math.log(self.val)
        return self.binary_operation(other, self.val**other.val, dfdx, dfdy)

    def max(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val if self.val >= other else other,
                1 if self.val >= other else 0)

        dfdx = 1 if self.val >= other.val else 0
        dfdy = 1 if self.val <= other.val else 0
        return self.binary_operation(other,
            self.val if self.val >= other.val else other.val, dfdx, dfdy)

    def min(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val if self.val <= other else other,
                1 if self.val <= other else 0)

        dfdx = 1 if self.val <= other.val else 0
        dfdy = 1 if self.val >= other.val else 0
        return self.binary_operation(other,
            self.val if self.val <= other.val else other.val, dfdx, dfdy)

    def exp(self):
        result = math.exp(self.val)
        return self.unary_operation(result, result)

    def log(self):
        return self.unary_operation(math.log(self.val), 1 / self.val)

    def log2(self):
        return self.unary_operation(math.log2(self.val), 1/(self.val * math.log(2)))

    def sin(self):
        return self.unary_operation(math.sin(self.val), math.cos(self.val))

    def cos(self):
        return self.unary_operation(math.cos(self.val), -math.sin(self.val))

    def tan(self):
        return self.unary_operation(math.tan(self.val), 1/(math.cos(self.val)**2))

    def __neg__(self):
        return self.unary_operation(-self.val, -1)

    # Make comparisons works like they do with numbers
    def __eq__(self, other):
        if type(other) != Node:
            return self.val == other
        return self.val == other.val

    def __ne__(self, other):
        if type(other) != Node:
            return self.val != other
        return self.val != other.val

    def __gt__(self, other):
        if type(other) != Node:
            return self.val > other
        return self.val > other.val

    def __lt__(self, other):
        if type(other) != Node:
            return self.val < other
        return self.val < other.val

    def __ge__(self, other):
        if type(other) != Node:
            return self.val >= other
        return self.val >= other.val

    def __le__(self, other):
        if type(other) != Node:
            return self.val <= other
        return self.val <= other.val

# Make function calls more intuitive and beautiful
def max(x, y):
    return x.max(y)

def min(x, y):
    return x.min(y)

def exp(x):
    return x.exp()

def log(x):
    return x.log()

def log2(x):
    return x.log2()

def sin(x):
    return x.sin()

def cos(x):
    return x.cos()

def tan(x):
    return x.tan()

# Use topo sort to filled up derivatives in computational graph from f
# Computational graph is a DAG
def derivative(f):
    # Init all derivatives and out degrees to 0, O(n)
    grads, out_degrees = [], []
    for i in range(f.index + 1):
        grads.append(0)
        out_degrees.append(0)
    # df/df = 1
    grads[f.index] = 1
    # Set out degrees, O(e)
    q = [f]
    while len(q) > 0:
        node = q.pop(0)
        for e_index in node.in_edge_indices:
            ch_index = edges[e_index]["src"]
            out_degrees[ch_index] += 1
            q.append(nodes[ch_index])
    # Topo sort, O(e)
    q = [f]
    while len(q) > 0:
        node = q.pop(0)
        for e_index in node.in_edge_indices:
            ch_index = edges[e_index]["src"]
            grads[ch_index] += grads[node.index] * edges[e_index]["dval"]
            out_degrees[ch_index] -= 1
            if out_degrees[ch_index] == 0:
                q.append(nodes[ch_index])
    return grads

# Test the automatic differentiation

# 1. y = log(x1) + x1 * x2 - sin(x2) + 1
# c = ln(x1)
# d = x1 * x2
# e = sin(x2)
# f = c + d
# g = f + e
# y = g + 1
# dy/dx1 = dy/dg * dg/df * (df/dc * dc/x1 + df/dd * dd/dx1)
# dy/dx2 = dy/dg * (dg/df * df/dd * dd/dx2 + dg/de * de/dx2)

# 2. y = (x1 + x2) * (x2 + 1)
# c = x1 + x2
# d = x2 + 1
# y = c * d
# dy/dx1 = dy/dc * dc/dx1
# dy/dx2 = dy/dc * dc/dx2 + dy/dd * dd/dx2
def test(f, x1, x2):
    x1, x2 = Node(x1), Node(x2)
    y = f(x1, x2)
    print("x1 =", x1, "x2 =", x2, "y =", y)
    print("Auto Diff")
    grads = derivative(y)
    print("dy/dx1 =", grads[x1.index], "dy/dx2 =", grads[x2.index])
    print("Numerical Diff")
    h = 0.00001
    print("dy/dx1 = ", (f(x1+h, x2) - f(x1-h, x2)) / (h * 2), 
        "dy/dx2 = ", (f(x1, x2+h) - f(x1, x2-h)) / (h * 2))

def main():
    print("y = ln(x1) + x1*x2 - sin(x2) + 1")
    test(lambda x1, x2: log(x1) + x1 * x2 - sin(x2) + 1, 2, 5)
    print("\ny = (x1 + x2) * (x2 + 1)")
    test(lambda x1, x2: (x1 + x2) * (x2 + 1), 2, 1)

if __name__ == "__main__":
    main()