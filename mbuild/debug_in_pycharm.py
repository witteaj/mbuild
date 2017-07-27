import mbuild as mb
from mbuild.recipes import Polymer
class AlkylMonomer(mb.Compound):
    """Creates a monomer for an alkyl chain"""
    def __init__(self):
        super(AlkylMonomer, self).__init__()
        self.add(mb.Particle(name='C', pos=[0,0,0]), label='C[$]')
        self.add(mb.Particle(name='Br', pos=[-0.109,0,0]), label='BrC[$]')
        self.add(mb.Particle(name='H', pos=[0.109,0,0]), label = 'HC[$]')
        theta = 0.5 * (180 - 109.5) * np.pi / 180
        self['BrC'][0].rotate(theta, around=[0,1,0])
        self['HC'][0].rotate(-theta, around=[0,1,0])
        self.add_bond((self[0],self['BrC'][0]))
        self.add_bond((self[0],self['HC'][0]))
        self.add(mb.Port(anchor=self[0]), label='up')
        self['up'].translate([0, -0.154/2, 0])
        self['up'].rotate(theta, around=[1,0,0])
        self.add(mb.Port(anchor=self[0]), label='down')
        self['down'].translate([0, 0.154/2, 0])
        self['down'].rotate(np.pi, around=[0,1,0])
        self['down'].rotate(-theta, around=[1,0,0])

import itertools as it
import numpy as np
import mbuild as mb
from mbuild.coordinate_transform import force_overlap
from mbuild.utils.validation import assert_port_exists
from mbuild import clone
__all__ = ['Polymer']
class Polymer(mb.Compound):
    """Connect one or more components in a specified sequence.
    Parameters
    ----------
    monomers : mb.Compound or list of mb.Compound
        The compound(s) to replicate.
    n : int
        The number of times to replicate the sequence.
    sequence : str, optional, default='A'
        A string of characters where each unique character represents one
        repetition of a monomer. Characters in `sequence` are assigned to
        monomers in the order assigned by the built-in `sorted()`.
    port_labels : 2-tuple of strs, optional, default=('up', 'down')
        The names of the two ports to use to connect copies of proto.
    """
    def __init__(self, monomers, n, sequence='A', port_labels=('up', 'down')):
        if n < 1:
            raise ValueError('n must be 1 or more')
        super(Polymer, self).__init__()
        if isinstance(monomers, mb.Compound):
            monomers = (monomers,)
        for monomer in monomers:
            for label in port_labels:
                assert_port_exists(label, monomer)
        unique_seq_ids = sorted(set(sequence))
        if len(monomers) != len(unique_seq_ids):
            raise ValueError('Number of monomers passed to `Polymer` class must'
                             ' match number of unique entries in the specified'
                             ' sequence.')
        # 'A': monomer_1, 'B': monomer_2....
        seq_map = dict(zip(unique_seq_ids, monomers))
        last_part = None
        for n_added, seq_item in enumerate(it.cycle(sequence)):
            this_part = clone(seq_map[seq_item])
            self.add(this_part, 'monomer[$]')
            if last_part is None:
                first_part = this_part
            else:
                # Transform this part, such that it's bottom port is rotated
                # and translated to the last part's top port.
                force_overlap(this_part,
                              this_part.labels[port_labels[1]],
                              last_part.labels[port_labels[0]])
            last_part = this_part
            if n_added == n * len(sequence) - 1:
                break
        # Hoist the last part's top port to be the top port of the polymer.
        self.add(last_part.labels[port_labels[0]], port_labels[0], containment=False)
        # Hoist the first part's bottom port to be the bottom port of the polymer.
        self.add(first_part.labels[port_labels[1]], port_labels[1], containment=False)

class OH(mb.Compound):
    """Creates a hydroxyl group"""
    def __init__(self):
        super(OH,self).__init__()
        self.add(mb.Particle(name='OA', pos = [0,0,0]), label='O')
        self.add(mb.Particle(name='H', pos = [.096, 0, 0]), label='H')
        self.add_bond((self[0], self[1]))
        self.add(mb.Port(anchor=self[0], orientation=[-1,
            np.tan(72*np.pi/180), 0], separation=.135/2), label='up')
class COOH(mb.Compound):
    """Creates headgroup of a carboxylic acid"""
    def __init__(self, ester = None):
        super(COOH,self).__init__()
        self.add(mb.Particle(name='C'), label='C')
        self.add(mb.Particle(name='O', pos = [0, .123, 0]),label='O[$]')
        self.add_bond((self['C'],self['O'][0]))

        self.add(mb.Port(anchor=self[0],
            orientation=[-1,-np.tan(34.5*np.pi/180),0],
            separation=.132/2), label='up')

        if ester:
            self.add(mb.Particle(name='O'), label='O[$]')
            self['O'][1].translate([.15,0,0])
            theta = ((180-111) / 2) * np.pi / 180
            self['O'][1].rotate(-theta, around=[0,0,1])
            self.add_bond((self['C'],self['O'][1]))
            self.add(mb.Port(anchor=self['O'][1],
                orientation=[1,np.tan(52.5*np.pi/180),0],
                separation=.14/2), label='down')
        else:
            self.add(mb.Port(anchor=self[0],
                orientation=[1,-np.tan(32*np.pi/180), 0],
                separation=.132/2),label='down')
            self.add(OH(),label='OH')
            mb.force_overlap(move_this=self['OH'],
                    from_positions=self['OH']['up'],
                    to_positions=self['down'])

class Methane(mb.Compound):
    def __init__(self):
        super(Methane, self).__init__()
        carbon = mb.Particle(name='C')
        self.add(carbon)
        hydrogen = mb.Particle(name='H', pos=[0.1, 0, -0.07])
        self.add(hydrogen)
        self.add_bond((self[0], self[1]))
        self.add(mb.Particle(name='O', pos=[-0.1, 0, -0.07]))
        self.add(mb.Particle(name='F', pos=[0, 0.1, 0.07]))
        self.add(mb.Particle(name='N', pos=[0, -0.1, 0.07]))
        self.add_bond((self[0], self[2]))
        self.add_bond((self[0], self[3]))
        self.add_bond((self[0], self[4]))
from mbuild.lib.moieties.ch3 import CH3
class FFA(mb.Compound):
    """Creates a saturated free fatty acid of n carbons based on user
    input"""
    def __init__(self, chain_length, ester=True):
        super(FFA, self).__init__()

        if ester:
            self.add(COOH(ester=True), label='head')
        else:
            self.add(COOH(), label='head')

        tail = mb.Polymer(AlkylMonomer(), (chain_length - 2))
        self.add(tail, label='tail')
        mb.x_axis_transform(self['tail'], new_origin=self['tail'][0],
                point_on_x_axis=self['tail'][6],
                point_on_xy_plane=self['tail'][3])
        self.add(CH3(), label='tailcap')

        self['head'].rotate(np.pi, [0,1,0])
        self['head'].translate([-self['tail'][3].pos[0],
            self['tail'][3].pos[1], 0])
        mb.force_overlap(move_this=self['tailcap'],
                from_positions=self['tailcap']['up'],
                to_positions=self['tail']['up'])

        self['head']['up'].spin(-np.pi/2, self['head'][0].pos)
        mb.force_overlap(move_this=self['head'],
                from_positions=self['head']['up'],
                to_positions=self['tail']['down'])
        self.spin(np.pi/2, [0,1,0])

mm = Methane()
from copy import deepcopy
from warnings import warn
import numpy as np
import mbuild as mb
__all__ = ['Monolayer']
class Monolayer(mb.Compound):
    """A general monolayer recipe.
    Parameters
    ----------
    surface : mb.Compound
        Surface on which the monolayer will be built.
    chains : list of mb.Compounds
        The chains to be replicated and attached to the surface.
    fractions : list of floats
        The fractions of the pattern to be allocated to each chain.
    backfill : list of mb.Compound, optional, default=None
        If there are fewer chains than there are ports on the surface,
        copies of `backfill` will be used to fill the remaining ports.
    pattern : mb.Pattern, optional, default=mb.Random2DPattern
        An array of planar binding locations. If not provided, the entire
        surface will be filled with `chain`.
    tile_x : int, optional, default=1
        Number of times to replicate substrate in x-direction.
    tile_y : int, optional, default=1
        Number of times to replicate substrate in y-direction.
    """
    def __init__(self, surface, chains, fractions=None, backfill=None, pattern=None,
                 tile_x=1, tile_y=1, **kwargs):
        super(Monolayer, self).__init__()
        # Replicate the surface.
        tiled_compound = mb.TiledCompound(surface, n_tiles=(tile_x, tile_y, 1))
        self.add(tiled_compound, label='tiled_surface')
        if pattern is None:  # Fill the surface.
            pattern = mb.Random2DPattern(len(tiled_compound.referenced_ports()))
        if isinstance(chains, mb.Compound):
            chains = [chains]

        if fractions:
            fractions = list(fractions)
            if len(chains) != len(fractions):
                raise ValueError("Number of fractions does not match the number"
                                 " of chain types provided")
            n_chains = len(pattern.points)
            # Attach chains of each type to binding sites based on
            # respective fractions.
            for chain, fraction in zip(chains[:-1], fractions[:-1]):
                # Create sub-pattern for this chain type
                subpattern = deepcopy(pattern)
                n_points = int(round(fraction * n_chains))
                warn("\n Adding {} of chain {}".format(n_points, chain))
                pick = np.random.choice(subpattern.points.shape[0], n_points,
                                        replace=False)
                points = subpattern.points[pick]
                subpattern.points = points
                # Remove now-occupied points from overall pattern
                pattern.points = np.array([point for point in pattern.points.tolist()
                                           if point not in subpattern.points.tolist()])
                # Attach chains to the surface
                attached_chains, _ = subpattern.apply_to_compound(
                    guest=chain, host=self['tiled_surface'], backfill=None, **kwargs)
                self.add(attached_chains)
        else:
            warn("\n No fractions provided. Assuming a single chain type.")
        # Attach final chain type. Remaining sites get a backfill.
        warn("\n Adding {} of chain {}".format(len(pattern), chains[-1]))
        attached_chains, backfills = pattern.apply_to_compound(guest=chains[-1],
                         host=self['tiled_surface'], backfill=backfill, **kwargs)
        self.add(attached_chains)
        self.add(backfills)
ff = FFA(chain_length=6)
spacings = [.94123, .94123, .94123]
basis = {'mm' : [[0., 0., 0.]], 'ff' : [[.5, .5, .5]]}
mixed_latty = mb.Lattice(spacings, basis_atoms=basis, dimension=3)
mixdix = {"mm" : mm, "ff" : ff}
mixed_crystal = mixed_latty.populate(x=2,y=2,z=3, compound_dict= mixdix)
# mixed_crystal.save('mixed_lattice_ff_mm.mol2', overwrite = True)
mixed_crystal.mirror(override= True)
# mixed_crystal.save('mixed_lattice_ff_mm_reg_mirror.mol2', overwrite = True)
mixed_crystal.mirror(override= True)
# mixed_crystal.save("mixed_lattice_ff_mm_flipped_back.mol2", overwrite= True)
OG = mb.clone(mixed_crystal)
# mixed_crystal.mirror(child_chirality= True, looking_for=["Methane"], override= True)
# mixed_crystal.save("mixed_lattice_mm_ff_w_mm_chirality.mol2", overwrite=True)
mixed_crystal.mirror(child_chirality= True, looking_for= "COOH", override= True)
print(mixed_crystal.children)
# for ii in OG.children:
#     print(ii)
# for ii, jj in zip(OG.children.map, mixed_crystal.children.map):
#     print('tier1')
#     print(ii.name)
#     print(jj.name)
#     print('_________')
#     for kk, ll in zip(ii.children.map, jj.children.map):
#         print('tier2')
#         print(kk.name)
#         print(ll.name)
#         print('')
#         for tt, rr in zip(kk.children.map, ll.children.map):
#             print('tier3')
#             print(tt)
#             print(rr)
#             print(tt.n_particles)
#             print(rr.n_particles)
#             print(tt.children)
#             print(rr.children)

