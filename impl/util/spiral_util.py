import numpy as np
from random import choice
import pdb


def get_adj_trig(adjecency_matrices, faces, reference_mesh):
    Adj = []
    for x in adjecency_matrices:
        adj_x = []
        dx = x.todense()
        for i in range(x.shape[0]):
            adj_x.append(dx[i].nonzero()[1])
            Adj.append(adj_x)

    reference_mesh_faces = reference_mesh.f

    # triangles for reference mesh
    trigs_full = [[] for i in range(len(Adj[0]))]
    for t in reference_mesh_faces:
        u, v, w = t
        trigs_full[u].append((u, v, w))
        trigs_full[v].append((u, v, w))
        trigs_full[w].append((u, v, w))

    # triangles for downsampled faces
    triangles = [trigs_full]
    for i, T in enumerate(faces):
        trigs_downsampled = [[] for i in range(len(Adj[i + 1]))]
        for u, v, w in T:
            trigs_downsampled[int(u)].append((u, v, w))
            trigs_downsampled[int(v)].append((u, v, w))
            trigs_downsampled[int(w)].append((u, v, w))
        triangles.append(trigs_downsampled)

    return Adj, triangles


def generate_spirals(step_sizes, meshes, Adj, triangles, reference_points, dilation=None, random=False,
                     counter_clockwise=True, nb_stds=2):
    Adj_spirals = []
    for i in range(len(Adj)):
        mesh_vertices = meshes[i].v

        spirals = get_spirals(mesh_vertices, Adj[i], triangles[i], reference_points[i], n_steps=step_sizes[i],
                              padding='zero', counter_clockwise=counter_clockwise, random=random)
        Adj_spirals.append(spirals)

    # TODO dilated convolution
    if dilation:
        raise("Dilation not implemented")


    # Spiral lengths
    L = []
    for i in range(len(Adj_spirals)):
        L.append([])
        for j in range(len(Adj_spirals[i])):
            L[i].append(len(Adj_spirals[i][j]))
        L[i] = np.aray(L[i])
    spiral_sizes = []
    for i in range(len(L)):
        spiral_size = L[i].mean() + nb_stds * L[i].std()
        spiral_sizes.append(int(spiral_size))

    spirals_np = []
    for i in range(len(spiral_sizes)):
        spiral = np.zeros((1, len(Adj_spirals[i]) + 1, spiral_sizes[i])) -1
        for j in range(len(Adj_spirals[i])):
            spiral[0, :len(Adj_spirals[i][j])] = Adj_spirals[i][j][:spiral_sizes[i]]
        spirals_np.append(spiral)

    return spirals_np, spiral_sizes, Adj_spirals


def get_spirals(mesh_vertices, adj, triangles, reference_points, n_steps=1, padding='zero', counter_clockwise=True,
                random=False):
    '''
    triangles contains triangles for all vertices
    '''
    spirals = []

    if not random:
        heath_path = None
        dist = None
        for reference_point in reference_points:
            heat_path, dist = single_source_shortest_path(mesh_vertices, adj, reference_point, dist, heath_path)

        heat_source = reference_points

    for i in range(mesh_vertices.shape[0]):
        seen = set()
        seen.add(i)
        triangle_center = list(triangles[i])
        A = adj[i]
        spiral = [i]

        # Starting point (first degree of freedom)
        if not random:
            if i in heat_source:  # closes to neighbour
                shortest_dist = np.inf
                initial_vertex = None
                for neighbour in A:
                    d = np.sum(np.square(mesh_vertices[i] - mesh_vertices[neighbour]))
                    if d < shortest_dist:
                        shortest_dist = d
                        initial_vertex = neighbour
            else:
                initial_vertex = heat_path[i]

        else:
            initial_vertex = choice(A)

        # first ring
        if initial_vertex is not None:
            ring = [initial_vertex]
            seen.add(initial_vertex)
        else:
            ring = []
        while len(triangle_center) > 0 and initial_vertex is not None:
            current_vertex = ring[-1]
            current_triangle = [t for t in triangle_center if t in triangles[current_vertex]]
            if len(ring) == 1:
                orientation_0 = (current_triangle[0][0] == i and current_triangle[0][1] == current_vertex) or \
                                (current_triangle[0][1] == i and current_triangle[0][2] == current_vertex) or \
                                (current_triangle[0][2] == i and current_triangle[0][0] == current_vertex)
                if not counter_clockwise:
                    orientation_0 = not orientation_0

                # Orientation of second point, (counter-)clockwise (second degree of freedom)
                if len(current_triangle) >= 2:
                    # chose triangle that directs spiral counter_clockwise
                    # The third vertex in the triangle is the next vertex in the spiral
                    if orientation_0:
                        third = [p for p in current_triangle[0] if p != i and p != current_vertex][0]
                        triangle_center.remove(current_triangle[0])
                    else:
                        third = [p for p in current_triangle[1] if p != i and p != current_vertex][0]
                        triangle_center.remove(current_triangle[1])
                    ring.append(third)
                    seen.add(third)
                elif len(current_triangle) == 1:
                    # break if spiral hits boundary in first vertex
                    break
            else:
                # unique ordering on the rest of the vertices
                if len(current_triangle) >= 1:
                    # third vertex in triangle is next vertex in spiral
                    third = [p for p in current_triangle[0] if p != current_vertex and p != i][0]
                    # if we have seen that vertex before, we do not add it to the spiral
                    if third not in seen:
                        ring.append(third)
                        seen.add(third)
                    triangle_center.remove(current_triangle[0])

                elif len(current_triangle) == 0:
                    # stop if the spiral hits the boundary
                    break

        rev_i = len(ring)
        if initial_vertex is not None:
            v = initial_vertex

            if orientation_0 and len(ring) == 1:
                reverse_order = False
            else:
                reverse_order = True
        need_padding = False

        while len(triangle_center) > 0 and initial_vertex is not None:
            current_triangle = [t for t in triangle_center if t in triangles[v]]
            if len(current_triangle) != 1:
                break
            else:
                need_padding = True

            third = [p for p in current_triangle[0] if p != v and p != i][0]
            triangle_center.remove(current_triangle[0])
            if third not in seen:
                ring.insert(rev_i, third)
                if not reverse_order:
                    rev_i = len(ring)
                v = third

        # dummy vertex in first and second half of spiral, similar to zero padding in a 2d grid
        if need_padding:
            ring.insert(rev_i, -1)

        spiral += ring

        # next rings:
        for step in range(n_steps - 1):
            next_ring = set([])
            next_triangles = set([])
            if len(ring) == 0:
                break
            base_triangle = None
            initial_vertex = None

            # neighbours in next hop
            for w in ring:
                if w != -1:
                    for u in adj[w]:
                        if u not in seen:
                            next_ring.add(u)

            # Triangles that do not countain two outer ring nodes. Enables following the spiral ordering by discarding
            # already visited triangles and nodes
            for u in next_ring:
                for triangle in triangles[u]:
                    if len([x for x in triangle if x in seen]) == 1:
                        next_triangles.add(triangle)
                    elif ring[0] in triangle and ring[-1] in triangle:
                        base_triangle = triangle

            # Starting point in the second ring
            # third poin in the triangle that connects first and last point in the first ring with the second ring
            if base_triangle is not None:
                initial_vertex = [x for x in base_triangle if x != ring[0] and x != ring[-1]]
                # Reassure that initial point is appropriate for starting the spiral
                # -> Make sure that it is connected to at least on of the next candidate vertices
                if (len(list(next_triangles.intersection(set(triangle[initial_vertex[0]]))))) == 0:
                    initial_vertex = None

            # if no triangle exists (if the vertex is a dummy vertex,
            # or both the first and the last vertex are part of a boundary,
            # or if the initial vertex is not connected with the rest of the ring)
            # A relative point in the triangle that connects the first point with the second
            # or the second with the third and so on...
            #
            if initial_vertex is None:
                for r in range(len(ring) - 1):
                    if ring[r] != -1 and ring[r + 1] != -1:
                        triangle = [t for t in triangle[ring[r]] if t in triangle[ring[r + 1]]]
                        for t in triangle:
                            initial_vertex = [v for v in t if v not in seen]
                            # need to make sure that the next vertex is appropriate to start spiral ordering in the next ring
                            if len(initial_vertex) > 0 and len(
                                    list(next_triangles.intersection(set(triangle[initial_vertex[0]])))) > 0:
                                break
                            else:
                                initial_vertex = []

                        if len(initial_vertex) > 0 and len(
                                list(next_triangles.intersection(set(triangle[initial_vertex[0]])))) > 0:
                            break
                        else:
                            initial_vertex = []

            # rest as for the first ring:
            if initial_vertex is None:
                initial_vertex = []
            if len(initial_vertex) > 0:
                initial_vertex = initial_vertex[0]
                ring = [initial_vertex]
                seen.add(initial_vertex)
            else:
                initial_vertex = None
                ring = []

            while len(next_triangles) > 0 and initial_vertex is not None:
                current_vertex = ring[-1]
                current_triangle = list(next_triangles.intersection(set(triangle[current_vertex])))

                if len(ring) == 1:
                    try:
                        orientation_0 = (current_triangle[0][0] in seen and current_triangle[0][1] == current_vertex) or \
                                        (current_triangle[0][1] in seen and current_triangle[0][2] == current_vertex) or \
                                        (current_triangle[0][2] in seen and current_triangle[0][0] == current_vertex)
                    except:
                        pdb.set_trace()

                    if not counter_clockwise:
                        orientation_0 = not orientation_0

                    # orientation for the next ring
                    if len(current_triangle) >= 2:
                        # chose triangle that will direct spiral counter-clockwise
                        if orientation_0:
                            # third point in triangle is the next vertex in the spiral
                            third = [p for p in current_triangle[0] if p not in seen and p != current_vertex][0]
                            next_triangles.remove(current_triangle[0])
                        else:
                            third = [p for p in current_triangle[1] if p not in seen and p != current_vertex][0]
                            next_triangles.remove(current_triangle[1])
                        ring.append(third)
                        seen.add(third)

                    # stop if the spiral hits boundary
                    elif len(current_triangle) == 1:
                        break

                else:
                    # unique ordering for the rest of the points
                    if len(current_triangle) >= 1:
                        third = [p for p in current_triangle[0] if p != v and p not in seen]
                        next_triangles.remove(current_triangle[0])
                        if len(third) > 0:
                            third = third[0]
                            if third not in seen:
                                ring.append(third)
                                seen.add(third)
                        else:
                            break

            rev_i = len(ring)
            if initial_vertex is not None:
                v = initial_vertex
                if orientation_0 and len(ring) == 1:
                    reverse_order = False
                else:
                    reverse_order = True

            need_padding = False

            while len(next_triangles) > 0 and initial_vertex is not None:
                current_triangle = [t for t in next_triangles if t in triangle[v]]
                if len(current_triangle) != 1:
                    break
                else:
                    need_padding = True

                third = [p for p in current_triangle[0] if p != v and p not in seen]
                next_triangles.remove(current_triangle[0])
                if len(third) > 0:
                    third = third[0]
                    if third not in seen:
                        ring.insert(rev_i, third)
                        seen.add(third)
                    if not reverse_order:
                        rev_i = len(ring)
                    v = third

            if need_padding:
                ring.insert(rev_i, -1)

            spiral += ring
        spirals.append(spiral)
    return spirals


def distance(v, w):
    return np.sqrt(np.sum(np.square(v - w)))


def single_source_shortest_path(V, E, source, dist=None, prev=None):
    import heapq
    if dist is None:
        dist = [None for i in range(len(V))]
        prev = [None for i in range(len(V))]
    q = []
    seen = set()
    heapq.heappush(q, (0, source, None))
    while len(q) > 0 and len(seen) < len(V):
        d_, v, p = heapq.heappop(q)
        if v in seen:
            continue
        seen.add(v)
        prev[v] = p
        dist[v] = d_
        for w in E[v]:
            if w in seen:
                continue
            dw = d_ + distance(V[v], V[w])
            heapq.heappush(q, (dw, w, v))

    return prev, dist
