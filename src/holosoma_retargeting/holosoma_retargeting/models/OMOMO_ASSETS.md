# OMOMO object assets

The object meshes in the sibling object directories are the simulation-ready
OMOMO meshes distributed by the InterMimic project. They were copied without
geometry or coordinate-frame changes from
`isaacgym/src/intermimic/data/assets/objects/objects` at InterMimic commit
`60d6d6e0895a308ff8dc8f4c53af211b739cd5e7`.

The pre-existing `largebox/largebox.obj` is byte-identical to the corresponding
InterMimic asset. Checksums for all thirteen supported assets are recorded in
`holosoma_retargeting.data_utils.object_assets`.

InterMimic is distributed under the MIT License and builds these assets from
the OMOMO dataset. Keep the InterMimic and OMOMO citations when publishing
results produced with these assets, and review the upstream dataset terms
before redistributing them outside this repository.
