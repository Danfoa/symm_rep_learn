import functools
from collections import OrderedDict

import numpy as np
from escnn.group import directsum, Representation
from escnn.nn import FieldType


def isotypic_decomp_rep(rep: Representation) -> Representation:
    """Returns a representation in a "symmetry enabled basis" (a.k.a Isotypic Basis).

    Adapted from MorphoSymm: https://github.com/Danfoa/MorphoSymm

    Takes a representation with an arbitrary basis (i.e., arbitrary change of basis and an arbitrary order of
    irreducible representations in the escnn Representation) and returns a new representation in which the basis
    is changed to a "symmetry enabled basis" (a.k.a Isotypic Basis). That is a representation in which the
    vector space is decomposed into a direct sum of Isotypic Subspaces. Each Isotypic Subspace is a subspace of the
    original vector space with a subspace representation composed of multiplicities of a single irreducible
    representation. In oder words, each Isotypic Subspace is a subspace with a subgroup of symmetries of the original
    vector space's symmetry group.

    Args:
        rep (Representation): Input representation in any arbitrary basis.

    Returns: A `Representation` with a change of basis exposing an Isotypic Basis (a.k.a symmetry enabled basis).
        The instance of the representation contains an additional attribute `isotypic_subspaces` which is an
        `OrderedDict` of representations per each isotypic subspace. The keys are the active irreps' ids associated
        with each Isotypic subspace.
    """
    symm_group = rep.group
    potential_irreps = rep.group.irreps()
    isotypic_subspaces_indices = {irrep.id: [] for irrep in potential_irreps}

    for pot_irrep in potential_irreps:
        cur_dim = 0
        for rep_irrep_id in rep.irreps:
            rep_irrep = symm_group.irrep(*rep_irrep_id)
            if rep_irrep == pot_irrep:
                isotypic_subspaces_indices[rep_irrep_id].append(list(range(cur_dim, cur_dim + rep_irrep.size)))
            cur_dim += rep_irrep.size

    # Remove inactive Isotypic Spaces
    for irrep in potential_irreps:
        if len(isotypic_subspaces_indices[irrep.id]) == 0:
            del isotypic_subspaces_indices[irrep.id]

    # Each Isotypic Space will be indexed by the irrep it is associated with.
    active_isotypic_reps = {}
    for irrep_id, indices in isotypic_subspaces_indices.items():
        irrep = symm_group.irrep(*irrep_id)
        multiplicities = len(indices)
        active_isotypic_reps[irrep_id] = Representation(group=rep.group,
                                                        irreps=[irrep_id] * multiplicities,
                                                        name=f'IsoSubspace {irrep_id}',
                                                        change_of_basis=np.identity(irrep.size * multiplicities),
                                                        supported_nonlinearities=irrep.supported_nonlinearities)

    # Impose canonical order on the Isotypic Subspaces.
    # If the trivial representation is active it will be the first Isotypic Subspace.
    # Then sort by dimension of the space from smallest to largest.
    ordered_isotypic_reps = OrderedDict(sorted(active_isotypic_reps.items(), key=lambda item: item[1].size))
    if symm_group.trivial_representation.id in ordered_isotypic_reps.keys():
        ordered_isotypic_reps.move_to_end(symm_group.trivial_representation.id, last=False)

    # Required permutation to change the order of the irreps. So we obtain irreps of the same type consecutively.
    oneline_permutation = []
    for irrep_id, iso_rep in ordered_isotypic_reps.items():
        idx = isotypic_subspaces_indices[irrep_id]
        oneline_permutation.extend(idx)
    oneline_permutation = np.concatenate(oneline_permutation)
    P_in2iso = permutation_matrix(oneline_permutation)

    Q_iso = rep.change_of_basis @ P_in2iso.T
    rep_iso_basis = directsum(list(ordered_isotypic_reps.values()),
                              name=rep.name + '-Iso',
                              change_of_basis=Q_iso)

    iso_supported_nonlinearities = [iso_rep.supported_nonlinearities for iso_rep in ordered_isotypic_reps.values()]
    rep_iso_basis.supported_nonlinearities = functools.reduce(set.intersection, iso_supported_nonlinearities)
    rep_iso_basis.attributes['isotypic_reps'] = ordered_isotypic_reps

    return rep_iso_basis

def field_type_to_isotypic_basis(field_type: FieldType):
    from escnn.group import change_basis

    rep = field_type.representation
    # Organize the irreps such that we get: rep_ordered_irreps := Q (⊕_k (⊕_i^mk irrep_k)) Q^T
    rep_ordered_irreps = isotypic_decomp_rep(rep)
    # Get dictionary of irrep_id: (⊕_i^mk irrep_k)
    iso_subspaces_reps = rep_ordered_irreps.attributes['isotypic_reps']
    # Define a field type composed of the representations of each isotypic subspace
    new_field_type = FieldType(gspace=field_type.gspace,
                               representations=list(iso_subspaces_reps.values()))
    return new_field_type

def permutation_matrix(oneline_notation):
    """Generate a permutation matrix from its oneline notation."""
    d = len(oneline_notation)
    assert d == np.unique(oneline_notation).size, "oneline_notation must describe a non-defective permutation"
    P = np.zeros((d, d), dtype=int)
    P[range(d), np.abs(oneline_notation)] = 1
    return P
