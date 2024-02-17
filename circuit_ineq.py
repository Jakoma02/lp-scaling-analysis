import itertools as it
import glpk
import math
import tempfile
import scipy.sparse
import random
import sys

from sage.all import vector, matrix, block_matrix, Integer, \
    identity_matrix, Matroid, Matrix, Graph, BipartiteGraph, \
    graphs, DiGraph, QQ, RR, abs_symbolic, infinity, diagonal_matrix, exp, \
    zero_vector


def read_mps(filename, free=True, use_gurobi=False):
    """
    Reads a LP from a MPS file into Sage matrices and vectors.

    Parameters:
      filename:    path to the MPS file
      free:        whether the free MPS format should be used
      use_gurobi:  preparse the problem using gurobipy before loading

    Returns:
      c:      utility function vector
      A_ub:   matrix of inequality (<=) constraints
      b_ub:   rhs of inequality constraints
      A_eq:   matrix of equality constraints
      b_eq:   rhs of equality constraints
      bounds: a list of (lower, upper) inclusive bounds for every
                variable, `None` means unbounded (e.g. `(-5, None)`)_
      integrality: a list of booleans for every variable (`True` means integral)
    """

    # The tempfile might not be used, we put it here so that the context can be large enough
    with tempfile.NamedTemporaryFile("w", suffix=".mps") as gurobi_out_file:
        if use_gurobi:
            import gurobipy as gp  # Licence is not needed when not used

            # A NASTY WORKAROUND: GLPK interprets MPS a little differently than gurobi
            # and MIPLIB do. However, reading and writing the MPS file using gurobi
            # changes the format slightly and GLPK then interprets it correctly.
            # So, we first read an MPS using gurobi, write it back to an MPS
            # file and we then read the new MPS file using GLPK.
            model = gp.read(filename)

            lp = None
            model.write(gurobi_out_file.name)
            glpk_in_filename = gurobi_out_file.name
        else:
            glpk_in_filename = filename

        if free:
            glpk_format = glpk.GLPK.GLP_MPS_FILE
        else:
            glpk_format = glpk.GLPK.GLP_MPS_DECK
        lp = glpk.mpsread(glpk_in_filename, fmt=glpk_format)

    c_scipy, A_ub_scipy, b_ub_scipy, A_eq_scipy, b_eq_scipy, bounds_scipy, integrality = lp

    # Is there a nicer syntax?
    c = vector(c_scipy) if c_scipy is not None else None
    A_ub = matrix(A_ub_scipy.todok(), nrows=A_ub_scipy.shape[0],
            ncols=A_ub_scipy.shape[1]) if A_ub_scipy is not None else None
    b_ub = vector(QQ, b_ub_scipy) if b_ub_scipy is not None else None
    A_eq = matrix(QQ, A_eq_scipy.todok(), nrows=A_eq_scipy.shape[0],
            ncols=A_eq_scipy.shape[1]) if A_eq_scipy is not None else None
    b_eq = vector(QQ, b_eq_scipy) if b_eq_scipy is not None else None
    bounds = list(bounds_scipy) if bounds_scipy is not None else []

    return c, A_ub, b_ub, A_eq, b_eq, bounds, integrality


def to_coo_matrix(sage_A):
    """
    Converts a sage (sparse) matrix to a scipy coo_matrix.
    """
    d = sage_A.dict()

    rows = []
    cols = []
    data = []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)

    return scipy.sparse.coo_matrix((data, (rows, cols)), 
                                   shape=(sage_A.nrows(), sage_A.ncols()))


def write_mps(c=None, A=None, b=None, bounds=None, integrality=None, filename="lp.mps", free=True):
    """
    Writes the LP to a MPS file.

    A_eq, A_ub are sage matrices or `None`
    c, b_eq, b_ub are sage vectors or `None`
    bounds, integrality are lists
    filename is a strng
    """

    if free:
        glpk_format = glpk.GLPK.GLP_MPS_FILE
    else:
        glpk_format = glpk.GLPK.GLP_MPS_DECK

    glpk.mpswrite(
        c.numpy(),
        A_eq=to_coo_matrix(A),
        b_eq=b.numpy(),
        bounds=bounds,
        integrality=integrality,
        filename=filename,
        fmt=glpk_format,
    )


def convert_to_standard(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """
    Converts the LP to form max. c^T x for Ax = b, x >= 0.
    Both inputs and outputs are Sage matrices and vectors.

    TODO: Don't lose information about variable integrality
    """

    # Make sure all subparts are defined
    if A_ub is None:
        A_ub = matrix([])
    if b_ub is None:
        b_ub = vector([])
    if A_eq is None:
        A_eq = matrix([])
    if b_eq is None:
        b_eq = vector([])

    A_ub_extra_columns = []
    A_eq_extra_columns = []
    c_extra_elements = []

    extra_bounds = []

    # Make sure all variables are non-negative
    # by rewriting x = x+ - x-, x+ >= 0, x- >= 0
    for i, (lb, ub) in enumerate(bounds):
        # It should be safe to assume lb <= ub
        if (lb is None or lb < 0) and (ub is None or ub > 0):
            # This variable needs to be split
            if A_ub.ncols() != 0:
                A_ub_extra_columns.append(-A_ub[:, i])
            if A_eq.ncols() != 0:
                A_eq_extra_columns.append(-A_eq[:, i])
            c_extra_elements.append(-c[i])

            bounds[i] = (0, ub)  # Bound for positive part
            new_lb = None if lb is None else -lb
            extra_bounds.append((0, new_lb))  # Bound for negative part

        elif (lb is None or lb < 0) and ub <= 0:
            # The variable has non-positive values, it suffices to
            # change its sign
            if A_ub.ncols() != 0:
                A_ub[:, i] *= -1
            if A_eq.ncols() != 0:
                A_eq[:, i] *= -1
            c[i] *= -1

            # The swap below is intentional
            new_lb = None if ub is None else -ub
            new_ub = None if lb is None else -lb
            bounds[i] = (new_lb, new_ub)

    # Add the extra columns/elements to original matrices
    A_ub = block_matrix([A_ub] + A_ub_extra_columns, nrows=1, subdivide=False)
    A_eq = block_matrix([A_eq] + A_eq_extra_columns, nrows=1, subdivide=False)
    c = vector(c.list() + c_extra_elements)

    # Add inequalities for bounds
    var_count = len(bounds) + len(extra_bounds)  # Before adding slacks
    bound_vectors = []
    bound_right_sides = []

    for i, (lb, ub) in enumerate(it.chain(bounds, extra_bounds)):
        # The first entry specifies the vector length
        if lb != 0:
            # Use -x <= -y instead of x >= y
            bound_vector = vector(QQ, var_count, sparse=True)
            bound_vector[i] = -1
            rhs = -lb
            bound_vectors.append(bound_vector)
            bound_right_sides.append(rhs)

        if ub is not None:
            bound_vector = vector(QQ, var_count, sparse=True)
            bound_vector[i] = 1
            rhs = ub
            bound_vectors.append(bound_vector)
            bound_right_sides.append(rhs)

    # Construct a matrix of all the inequalities
    bound_matrix = matrix(bound_vectors)

    if bound_matrix.nrows() != 0:
        complete_ineq_mat = block_matrix([[A_ub, Integer(0)], [bound_matrix]],
                                         subdivide=False)
        complete_ineq_right_side = vector(b_ub.list() + bound_right_sides)
    else:
        complete_ineq_mat = matrix([])
        complete_ineq_right_side = vector([])

    slack_vars_count = complete_ineq_mat.nrows()
    slacked_ineq_mat = block_matrix(
        [[complete_ineq_mat, identity_matrix(slack_vars_count)]],
        subdivide=False
    )

    if slacked_ineq_mat.nrows() != 0:
        result_mat = block_matrix([[A_eq, Integer(0)], [slacked_ineq_mat]],
                                  subdivide=False)
    else:
        result_mat = A_eq
    result_b = concat_vectors(b_eq, complete_ineq_right_side)
    result_c = vector(c.list() + slack_vars_count * [0])

    return result_c, result_mat, result_b


def concat_vectors(u, v):
    """
    Returns a concatenated vector of u and v. This is meant mainly to be an abstraction
    so that it is possible to switch to a more efficient implementation in the future.
    """
    return vector(u.list() + v.list())


def convert_to_semi_standard(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality):
    """
    Converts the LP to "semi-standard" form: min c^T x for Ax = b,
    bounds[i][0] <= x[i] <= bounds[i][1] (if the bounds are not None).
    """
    # Make sure all subparts are defined
    if A_ub is None:
        A_ub = matrix([], sparse=True)
    if b_ub is None:
        b_ub = vector([], sparse=False)
    if A_eq is None:
        A_eq = matrix([], sparse=True)
    if b_eq is None:
        b_eq = vector([], sparse=False)

    slack_var_count = A_ub.nrows()
    slacked_ineq_mat = block_matrix(
        [
            [A_ub, identity_matrix(slack_var_count, sparse=True)]
        ],
        subdivide=False,
        sparse=True,
    )

    if slack_var_count != 0:
        result_matrix = block_matrix(
            [
                [A_eq, Integer(0)],
                [slacked_ineq_mat]
            ],
            subdivide=False,
            sparse=True,
        )
    else:
        result_matrix = A_eq

    result_b = concat_vectors(b_eq, b_ub)
    result_c = concat_vectors(c, zero_vector(slack_var_count))
    result_integrality = integrality + slack_var_count * [0]
    result_bounds = bounds + slack_var_count * [(0, infinity)]

    return result_c, result_matrix, result_b, result_bounds, result_integrality


def fund_circuit_graph(fund_circuits, basis, nonbasic):
    """
    Returns a graph where V = elements, E = {{b, j} | be is contained in fundamental circuit of j}.

    The graph is a bipartite graph (a slightly different representation than the one described in the paper).
    """
    # Only so that the vertices are positioned in two partites
    graph = BipartiteGraph()
    graph.add_vertices(nonbasic, left=False, right=True)
    graph.add_vertices(basis, left=True, right=False)

    for circ in fund_circuits:
        circ_nonbasic_elements = circ.difference(basis)
        # This is a little hacky. There should be exactly one element
        # in the difference, we return it by the next(iter(...)) construct.
        circ_nonbasic_element = next(iter(circ_nonbasic_elements))
        
        basic_elements = circ.difference(circ_nonbasic_elements)
        graph.add_edges(((circ_nonbasic_element, x, circ_nonbasic_element) for x in basic_elements))

    return graph


def ij_circuit_uv_sets(fund_circuit_graph, basis, i, j):
    r"""
    Returns a tuple (U, V) of sets from the paper s.t. (B \ U) u V
    contains a unique circuit containing both i and j
    """
    path = fund_circuit_graph.shortest_path(i, j)

    # Basic path ends don't belong to either U or V
    if i in basis:
        path = path[1:]
    if j in basis:
        path = path[:len(path) - 1]

    u = path[1::2]
    v = path[0::2]

    return u, v


# @profile
def representant_from_uv(A_rref, basis, u, v):
    r"""
    Finds a minimum-support vector in Ker(A_rref) s.t. its support is contained
    in (B \ U) u V
    """

    basis_list = sorted(list(basis))
    basis_set = set(basis)

    # Find indices of elements of u within the basis, all within one pass
    # (two pointers)
    u_basis_indices = []
    u.sort()
    j = 0
    for i, b in enumerate(basis_list):
        if j == len(u):
            break

        if b == u[j]:
            u_basis_indices.append(i)

            j += 1

    u_basis_indices_set = set(u_basis_indices)
    remaining_indices_set = set(range(A_rref.nrows())).difference(u_basis_indices_set)
    remaining_indices = sorted(list(remaining_indices_set))

    # Let's define W := B \ U
    w_set = basis_set.difference(u)
    w_list = sorted(list(w_set))

    v_list = sorted(list(v))

    s = basis_set.difference(u).union(v)  # The whole circuit superset
    s_list = list(s)

    new_vector = vector(QQ, A_rref.ncols())

    # It might happen that there are no vertices in U
    if u:
        # Rows of matrix correspond to basis indices -- given by RREF
        # We are only interested in columns of S
        u_submatrix = A_rref[u_basis_indices, s_list]

        graph = _vector_finding_graph(u_submatrix)
        first_vertex = graph.vertices(sort=True)[0]

        # Indirect indexing -- we want to index into the big
        # vector using indices within s
        new_vector[s_list[first_vertex]] = 1

        dfs_edges = graph.depth_first_search(first_vertex, edges=True)
        for u1, u2 in dfs_edges:
            l = graph.edge_label(u1, u2)
            new_vector[s_list[u2]] = new_vector[s_list[u1]] * l
    else:
        # set the only element from V to 1
        new_vector[v[0]] = 1

    # Product before setting values of entries from W.
    # We can restrict ourself to looking at rows of W
    # and columns of V.
    remaining_v_submatrix = A_rref[remaining_indices, v_list]
    
    # The vector <-> matrix conversions are a hack to allow indexing of columns by a set
    v_new_subvector = vector(Matrix(new_vector)[:,v_list]) 
    early_product = remaining_v_submatrix * v_new_subvector

    # Set the other values so that it is in the kernel
    for i, x in enumerate(early_product):
        # assert x == 0 or new_vector[w_list[i]] == 0 or -x == new_vector[w_list[i]]
        new_vector[w_list[i]] = -x

     
    # assert A_rref * new_vector == 0
    return new_vector


def _vector_finding_graph(A):
    """
    Builds a "vector-finding" digraph for the given matrix.

    Vertices are columns and there is an edge (u, v, l)
    iff there is a row that has exactly two non-zeros a, b
    at positions u, v and it holds l=-a/b.
    The meaning is as follows: if var. `u` has value `x`,
    then variable `v` must have value `l * x`.

    The function supposes that every row of the matrix has 0
    or exactly 2 non-zeros.
    """
    A_numpy = A.numpy()
    nonzero_rows, nonzero_cols = A_numpy.nonzero()
    nonzero_vals = [A[r,c] for r,c in zip(nonzero_rows, nonzero_cols)]
    
    rows_indices = [[] for _ in range(A_numpy.shape[0])]

    for r, c, val in zip(nonzero_rows, nonzero_cols, nonzero_vals):
        rows_indices[r].append((c, val))

    nonempty_row_indices = list(map(tuple, filter(None, rows_indices)))

    # We need directed edges in both directions
    edges = [(c1, c2, -v1/v2) for ((c1, v1), (c2, v2)) in nonempty_row_indices]
    edges.extend([(c2, c1, -v2/v1) for ((c1, v1), (c2, v2)) in nonempty_row_indices])

    g = DiGraph()
    g.add_edges(edges)
    return g


def vector_pairwise_kappa(vec, i, j):
    """
    Returns a single number -- "kappa" for the single given vector
    (not a minimum) and given i, j.
    
    kappa = |val_j / val_i|
    """
    
    # Symbolic division
    return abs_symbolic(vec[j] / vec[i])


def min_cycle_mean(graph, can_modify=False, must_relabel=True):
    """
    Finds the minimum cycle mean weight in a directed
    strongly connected graph.

    Current implementation uses Karp's algorithm.

    `can_modify` allows to modify the input graph in-place
    """

    # Number the vertices so that we can use then as indices
    if must_relabel:
        if can_modify:
            graph.relabel(None)
        else:
            graph = graph.relabel(None, inplace=False)

        n = graph.num_verts()

    # dp[v][k] from Karp's algorithm -- minimum weight of edge
    # progression from s to v of length exactly k
    dp = [[infinity for _ in range(n + 1)] for _ in range(n)]

    s = graph.vertices(sort=True)[0]
    dp[s][0] = 0

    for k in range(1, n + 1):
        for v in range(n):
            if graph.in_degree(v) == 0:
                continue
            dp[v][k] = min((dp[e[0]][k - 1] + e[2] for e in graph.incoming_edge_iterator(v)))

    v_maxima = ((max(((dp[v][n] - dp[v][k]) / (n - k) for k in range(n) if dp[v][k] < infinity), default=infinity)) for v in range(n))

    min_cycle_mean = min(v_maxima)

    return min_cycle_mean


def max_cycle_mean(graph, must_relabel=True):
    """
    Finds the weight of the maximum mean cycle.

    `must_relabel == False` means that vertices are already numbered 0 to (N-1)

    Use the min-mean cycle as a subprecedure.
    """
    neg_graph = graph.copy()

    for (u, v, l) in neg_graph.edges(sort=False):
        neg_graph.set_edge_label(u, v, -l)

    return -1 * min_cycle_mean(neg_graph, can_modify=True, must_relabel=must_relabel)


def log_pairwise_kappa_digraph(pw_kappas):
    """
    Construct a complete DiGraph of log pairwise kappas
    as described in the article.

    pw_kappas[i][j] is kappa_ij
    """
    n = len(pw_kappas)
    edges = [(i, j, math.log(pw_kappas[i][j])) for i, j in it.permutations(range(n), r=2)]
    g = DiGraph(edges, weighted=True)

    return g


def representant_matrix(A_rref):
    """
    Return a tuple (representants, index_matrix)
    s.t. representants[index_matrix[i][j]] is a minimum-support
    vector from Ker(A_rref) s.t. its support contains i and j.
    Values on the diagonal are not defined.

    The matroid corresponding to A_rref must be irreducible for this
    function to work properly.
    """
    n = A_rref.ncols()
    representants = []
    index_matrix = [[None for _ in range(n)] for _ in range(n)]

    A_rref_matroid = Matroid(A_rref)

    basis = A_rref.pivots()
    nonbasis = A_rref_matroid.groundset().difference(basis)

    fund_circuits = [A_rref_matroid.fundamental_circuit(basis, x) for x in nonbasis]
    fund_circuit_g = fund_circuit_graph(fund_circuits, basis, nonbasis)

    for i, j in it.permutations(range(n), r=2):
        if index_matrix[i][j] is not None:
            continue

        u, v = ij_circuit_uv_sets(fund_circuit_g, basis, i, j)
        representant = representant_from_uv(A_rref, basis, u, v)
        circuit = representant.support()

        representants.append(representant)
        representant_index = len(representants) - 1

        for k, l in it.permutations(circuit, r=2):
            pw_kappa = vector_pairwise_kappa(representant, k, l)
            if index_matrix[k][l] is None or vector_pairwise_kappa(representants[index_matrix[k][l]], k, l) < pw_kappa:
                index_matrix[k][l] = representant_index

    return representants, index_matrix


def _pseudokappa_dig(kappa_dig, log_kappa_star_estimate):
    # A copy might not be needed
    dig = kappa_dig.copy()
    for (a, b, l) in dig.edge_iterator():
        # the epsilon is needed to avoid negative cycles
        epsilon = 1e-10
        dig.set_edge_label(a, b, log_kappa_star_estimate - l + epsilon)

    n = kappa_dig.order()
    r = dig.add_vertex()
    extra_edges = [(r, x, 0) for x in range(n)]
    dig.add_edges(extra_edges)

    return dig, r


def random_d_vector(A_rref):
    n = A_rref.ncols()

    ds = []
    for _ in range(n):
        r = random.randint(100, 100000)
        r_float = r / 100
        above_zero = random.randint(0, 1)
        if above_zero == 0:
            r_float = 1 / r_float

        ds.append(r_float)

    return ds


def rescale_vector(vec, ds):
    return [d * x for (d, x) in zip(ds, vec)]


def kappa_estimate_for_given_representants(representants, index_matrix):
    """
    Estimate kappa as a maximum over pairwise kappas FOR THEIR RESPECTIVE REPRESENTANTS
    (not necessarily the best ones from the representant list).
    """
    estimate = -infinity

    for i, j in it.product(range(len(index_matrix)), repeat=2):
        if i == j:
            continue
        representant = representants[index_matrix[i][j]]
        representant_pairwise_kappa = vector_pairwise_kappa(representant, i, j)
        estimate = max(estimate, representant_pairwise_kappa)

    return estimate


def fail_sanity(msg):
    print(f"SANITY CHECK FAILED: {msg}", file=sys.stderr)
    exit(1)


def check_dvector_sanity(A_rref, representants, index_matrix, ds, original_kappa_estimate, new_kappa_estimate):
    RANDOM_PRESCALINGS = 100

    print("Starting sanity check...")
    representants = [tuple((float(x) for x in rep)) for rep in representants]

    # Check that the new kappa estimate is better (that is correct only when considering the same representants)
    if original_kappa_estimate < new_kappa_estimate:
        fail_sanity(f"Original kappa estimate ({original_kappa_estimate}) as better than "
                    f"the new one ({new_kappa_estimate})")

    # Check that the original kappa estimate is correct
    vector_kappas = [float(vector_kappa(rep)) for rep in representants]
    kappa_estimate = max(vector_kappas)

    if not math.isclose(kappa_estimate, original_kappa_estimate):
        fail_sanity(f"We claim original kappa can be estimated by {original_kappa_estimate}, "
                    f"but it fact it can only be estimated by {kappa_estimate}")

    # Check that the new kappa estimate is max pairwise kappa over representants
    rescaled_representants = [rescale_vector(representant, ds) for representant in representants]
    correct_new_estimate = kappa_estimate_for_given_representants(rescaled_representants, index_matrix)

    if not math.isclose(new_kappa_estimate, correct_new_estimate):
        fail_sanity(f"We claim new kappa estimate is {new_kappa_estimate}, but by pairwise kappas "
                    f"over representants the correct estimate is {correct_new_estimate}")

    # Check new is better than any random for every representant
    for _ in range(RANDOM_PRESCALINGS):
        random_ds = random_d_vector(A_rref)
        rescaled_representants = [rescale_vector(representant, random_ds) for representant in representants]
        rescaled_estimate = kappa_estimate_for_given_representants(rescaled_representants, index_matrix)

        if rescaled_estimate < new_kappa_estimate:
            fail_sanity(f"A random prescaling gave better estimate kappa estimate ({kappa_estimate}) "
                        f"than the optimal one (with respect to selected representants) ({new_kappa_estimate})")


def _component_dvector_estimate(component_A_rref, check_sanity=False):
    """
    Finds a nearly optimal vector d s.t. the diagonal
    matrix D with entries from d has minimal kappa(A_rref * D)
    together with original matrix kappa estimate and rescaled
    matrix kappa estimate.

    The matrix matroid of A_rref must be irreducible --
    holds trivially when A_rref is a single component
    of the matroid.
    """
    representants, index_matrix = representant_matrix(component_A_rref)
    original_kappa_estimate = float(max((vector_kappa(r) for r in representants)))

    n = component_A_rref.ncols()

    kappas = [[None for _ in range(n)] for _ in range(n)]
    for i, j in it.combinations(range(n), 2):
        # Stop doing symbolic calculations at this point
        kappas[i][j] = float(vector_pairwise_kappa(representants[index_matrix[i][j]], i, j))
        kappas[j][i] = float(1 / kappas[i][j])

    kappa_dig = log_pairwise_kappa_digraph(kappas)
    log_kappa_star_estimate = max_cycle_mean(kappa_dig)

    new_kappa_estimate = exp(log_kappa_star_estimate)

    pseudokappa_g, r = _pseudokappa_dig(kappa_dig, log_kappa_star_estimate)
    path_len_dict = pseudokappa_g.shortest_path_lengths(r, by_weight=True, algorithm="Bellman-Ford_Boost")
    path_lenghts = (path_len_dict[i] for i in range(n))

    ds = [math.exp(l) for l in path_lenghts]

    # We need to take 1/d, because if W = Ker(A), WD = Ker(1/d A)
    inv_ds = [1/d for d in ds]
    d_vector = vector(inv_ds)

    if check_sanity:
        check_dvector_sanity(component_A_rref, representants, index_matrix,
                             ds, original_kappa_estimate, new_kappa_estimate)

    return d_vector, original_kappa_estimate, new_kappa_estimate


def estimate_optimal_preconditioning_ds(A, check_sanity=False):
    """
    Returns a nearly optimal diagonal non-negative matrix D
    s.t. kappa(AD) is minimal together with an estimate of
    kappa of the original program (from the circuits we find)
    and an estimate of kappa of the rescaled program.
    """
    A_rat = Matrix(QQ, A)  # So that we have exact arithmetic
                           # and nothing can break
    A_matroid = Matroid(A_rat)

    total_original_kappa_estimate = 0
    total_new_kappa_estimate = 0

    # By default prescale columns by 1
    preconditioner_diagonal = vector(QQ, A_rat.ncols(), A_rat.ncols() * [1])

    for component in A_matroid.components():
        component_list = list(component)

        # Including zero rows (that would be separate components)
        component_A = A_rat[:,component_list]
        component_A_rref = component_A.rref()

        # Remove the zero rows
        nonzero_row_list, _ = component_A_rref.numpy().nonzero()
        uniq_nonzero_row_list = list(set(nonzero_row_list))

        component_A_rref = component_A_rref[uniq_nonzero_row_list,:]

        if component_A_rref.rank() == 0:
            # A weird case when the matrix only contains zero columns
            continue

        if component_A_rref.rank() == component_A_rref.ncols():
            # There are no circuits, we can skip this component
            continue
        
        component_d, original_kappa_estimate, new_kappa_estimate = _component_dvector_estimate(component_A_rref, check_sanity)
        # We subtract one from every d so that when added with the
        # initial one we get the desired value
        corrected_component_d = component_d - vector(len(component_d) * [1])

        total_original_kappa_estimate = max(total_original_kappa_estimate, original_kappa_estimate)
        total_new_kappa_estimate = max(total_new_kappa_estimate, new_kappa_estimate)

        # Lift the component solution back to the space of all variables
        I_submatrix = identity_matrix(A.ncols())[:,component_list]

        preconditioner_diagonal += I_submatrix * corrected_component_d

    return preconditioner_diagonal, total_original_kappa_estimate, total_new_kappa_estimate


def to_preconditioned_mip(c, A, b, bounds, integrality, ds):
    """
    Returns a MIP equivalent to the original one by prescaling
    matrix columns by a diagonal matrix with elements of `ds`
    on the diagonal.
    """
    D = diagonal_matrix(ds)

    inv_ds = [1/d for d in ds]
    inv_D = diagonal_matrix(inv_ds)

    rescaled_A = A * D

    left_bounds_v = [bnd[0] for bnd in bounds]
    right_bounds_v = [bnd[1] for bnd in bounds]

    rescaled_left_bounds = [di * bnd for (di, bnd) in zip(inv_ds, left_bounds_v)]
    rescaled_right_bounds = [di * bnd for (di, bnd) in zip(inv_ds, right_bounds_v)]
    rescaled_bounds = list(zip(rescaled_left_bounds, rescaled_right_bounds))

    new_bounds = []
    for (bound, d) in zip(bounds, ds):
        l, u = bound
        # Should handle infinity properly
        new_l = l/d
        new_u = u/d

        new_bounds.append((new_l, new_u))

    integral_indices = [i for i, x in enumerate(integrality) if x]

    left_part = -D[integral_indices,:]
    right_part = identity_matrix(len(integral_indices))

    new_A = block_matrix(
        [
            [rescaled_A, Integer(0)],
            [left_part, right_part],
        ],
        subdivide=False
    )

    rescaled_c = diagonal_matrix(ds) * c
    new_integrality = len(rescaled_bounds) * [0] + len(integral_indices) * [1]
    new_bounds = rescaled_bounds + len(integral_indices) * [(-infinity, infinity)]
    new_c = vector(list(rescaled_c) + len(integral_indices) * [0])
    new_b = vector(list(b) + len(integral_indices) * [0])

    return new_c, new_A, new_b, new_bounds, new_integrality


def witness_vector(A, circuit):
    """
    This function finds a vector in the right kernel of A
    s.t. the support of the vector is the given circuit.
    Such vector is unique up to rescaling.

    This is not meant to be fast and is used when computing
    exact kappa (slow).
    """
    circ_list = list(circuit)
    submatrix = A[:,circ_list]
    r_kernel = submatrix.right_kernel_matrix()

    assert(r_kernel.nrows() == 1)
    return vector(r_kernel)


def vector_kappa(v):
    """
    Returns a kappa corresponding to given vector
    (the real kappa is a maximum over all elementary vectors).
    """
    abs_vector = [abs(x) for x in v if x != 0]
    return max(abs_vector) / min(abs_vector)


def exact_kappa(A):
    """
    Calculates the exact value of kappa for the matrix A.
    This is slow (known to be NP hard) and only feasible for
    small matrices.
    """
    MAX_ERROR = 1e-4
    if A.base_ring() != QQ:
        RR_A = Matrix(RR, A)
        approx_A = RR_A.apply_map(lambda x: x.nearby_rational(MAX_ERROR))
        A_rat = Matrix(QQ, approx_A)
    else:
        A_rat = A
    # This needs to be done because sage cannot
    # work with matroids from matrices with real
    # entries
    M = Matroid(A_rat)

    max_kappa = 0
    for c in M.circuits():
        wv = witness_vector(A_rat, c)
        max_kappa = max(max_kappa, vector_kappa(wv))

    return max_kappa


def lp_relaxation(c, A, b, bounds, integrality):
    new_integrality = len(integrality) * [0]
    return c, A, b, bounds, new_integrality
