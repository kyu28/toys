import math

# Computational Graph
# Calculate the derivative while evaluating the expression
# Use chain rule to get the derivative by calling derivative(f, x)
nodes = []

class Edge:
    src, dest, dval = 0, 0, 0
    def __init__(self, src, dest, dval):
        self.src, self.dest, self.dval = src, dest, dval

class Node:
    # The Number, the product of the derivative, the index in nodes list
    val, deriv_prod, index = None, 0, None
    # The edges connected to this node
    src = [] 

    # Generate result node and connect computational graph
    def unary_operation(self, result, dfdx):
        result = Node(result)
        result.src = [Edge(self.index, result.index, dfdx)]
        nodes.append(result)
        return result

    def binary_operation(self, other, result, dfdx, dfdy):
        result = Node(result)
        result.src = [
            Edge(self.index, result.index, dfdx),
            Edge(other.index, result.index, dfdy)]
        nodes.append(result)
        return result

    def __init__(self, val):
        self.val = val
        self.index = len(nodes)
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
        return self.__mul__(other)

    def __div__(self, other):
        if type(other) != Node:
            return self.unary_operation(self.val/other, 1/other)
        return self.binary_operation(other, self.val/other.val, 1/other.val,
            -self.val / (other.val**2))

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

# The real trick
def derivative(f, x):
    for node in nodes:
        node.deriv_prod = 0
    f.deriv_prod = 1
    q = [f]
    while len(q) != 0:
        n = q.pop(0)
        for edge in n.src:
            child = nodes[edge.src]
            child.deriv_prod += n.deriv_prod * edge.dval
            q.append(child)
    return x.deriv_prod


# Test the automatic differentiation

# c = ln(x1)
# d = x1 * x2
# e = sin(x2)
# f = c + d
# g = f + e
# y = g + 1
# dy/dx1 = dy/dg * dg/df * (df/dc * dc/x1 + df/dd * dd/dx1)
# dy/dx2 = dy/dg * (dg/df * df/dd * dd/dx2 + dg/de * de/dx2)
def test1():
    print("y = ln(x1) + x1*x2 - sin(x2) + 1")
    x1, x2 = Node(2), Node(5)
    y = log(x1) + x1 * x2 - sin(x2) + 1
    print("x1 = ", x1, "dy/dx1 = ", derivative(y, x1)) # Should be 5.5
    print("x2 = ", x2, "dy/dx2 = ", derivative(y, x2)) # Should be 1.716

# c = a + b
# d = b + 1
# e = c * d
# de/da = de/dc * dc/da
# de/db = de/dc * dc/db + dd/db
def test2():
    print("e = (a + b) * (b + 1)")
    a, b = Node(2), Node(1)
    e = (a + b) * (b + 1)
    print("a = ", a, "de/da = ", derivative(e, a)) # Should be 2
    print("b = ", b, "de/db = ", derivative(e, b)) # Should be 5

def main():
    test1()
    print()
    test2()

if __name__ == "__main__":
    main()
