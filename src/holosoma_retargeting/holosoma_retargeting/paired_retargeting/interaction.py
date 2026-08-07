# ruff: noqa: CPY001
"""Pure cross-actor interaction-mesh construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, QhullError


@dataclass(frozen=True)
class CrossActorInteractionMesh:
    """A bipartite Laplacian graph extracted from a Delaunay mesh."""

    tetrahedra: np.ndarray
    edges: np.ndarray
    laplacian_matrix: np.ndarray


def _validated_actor_vertices(name: str, vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _cross_actor_tetrahedra(vertices: np.ndarray, actor_a_count: int) -> np.ndarray:
    try:
        tetrahedra = np.asarray(Delaunay(vertices).simplices, dtype=np.int32)
    except QhullError as exc:
        raise ValueError("Cross-actor points do not form a three-dimensional interaction mesh") from exc
    return tetrahedra[np.any(tetrahedra < actor_a_count, axis=1) & np.any(tetrahedra >= actor_a_count, axis=1)]


def _cross_actor_edges(tetrahedra: np.ndarray, actor_a_count: int) -> np.ndarray:
    edges = {
        tuple(sorted((int(source), int(target))))
        for tetrahedron in tetrahedra
        for source in tetrahedron
        for target in tetrahedron
        if (source < actor_a_count) != (target < actor_a_count)
    }
    if not edges:
        raise ValueError("Interaction mesh contains no cross-actor edges")
    return np.asarray(sorted(edges), dtype=np.int32)


def _laplacian_matrix(vertex_count: int, edges: np.ndarray) -> np.ndarray:
    neighbors: list[list[int]] = [[] for _ in range(vertex_count)]
    for source, target in edges:
        neighbors[source].append(int(target))
        neighbors[target].append(int(source))
    laplacian = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    for vertex_index, adjacent in enumerate(neighbors):
        if adjacent:
            laplacian[vertex_index, vertex_index] = 1.0
            laplacian[vertex_index, adjacent] = -1.0 / len(adjacent)
    return laplacian


def create_cross_actor_interaction_mesh(
    actor_a_vertices: np.ndarray,
    actor_b_vertices: np.ndarray,
) -> CrossActorInteractionMesh:
    """Create a Delaunay mesh and retain only graph edges crossing actors."""
    actor_a = _validated_actor_vertices("actor_a_vertices", actor_a_vertices)
    actor_b = _validated_actor_vertices("actor_b_vertices", actor_b_vertices)
    vertices = np.vstack((actor_a, actor_b))
    actor_a_count = actor_a.shape[0]
    cross_tetrahedra = _cross_actor_tetrahedra(vertices, actor_a_count)
    edge_array = _cross_actor_edges(cross_tetrahedra, actor_a_count)
    return CrossActorInteractionMesh(
        tetrahedra=cross_tetrahedra,
        edges=edge_array,
        laplacian_matrix=_laplacian_matrix(vertices.shape[0], edge_array),
    )


def laplacian_coordinates(mesh: CrossActorInteractionMesh, vertices: np.ndarray) -> np.ndarray:
    """Evaluate mesh Laplacian coordinates for a vertex array."""
    points = np.asarray(vertices, dtype=np.float64)
    expected_shape = (mesh.laplacian_matrix.shape[0], 3)
    if points.shape != expected_shape:
        raise ValueError(f"Interaction vertices must have shape {expected_shape}, got {points.shape}")
    return mesh.laplacian_matrix @ points
