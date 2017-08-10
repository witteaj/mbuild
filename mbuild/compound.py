from __future__ import print_function, division

__all__ = ['load', 'clone', 'Compound', 'Particle']

import collections
from collections import OrderedDict, defaultdict
from copy import deepcopy
import itertools
import os
import sys
import tempfile
from warnings import warn

import mdtraj as md
import numpy as np
from oset import oset as OrderedSet
import parmed as pmd
from parmed.periodic_table import AtomicNum, element_by_name, Mass
import simtk.openmm.app.element as elem
from six import integer_types, string_types

from mbuild.bond_graph import BondGraph
from mbuild.box import Box
from mbuild.exceptions import MBuildError
from mbuild.formats.hoomdxml import write_hoomdxml
from mbuild.formats.lammpsdata import write_lammpsdata
from mbuild.formats.gsdwriter import write_gsd
from mbuild.periodic_kdtree import PeriodicCKDTree
from mbuild.utils.io import run_from_ipython, import_
from mbuild.coordinate_transform import _translate, _rotate, normalized_matrix, angle, unit_vector
import mbuild as mb



def load(filename, relative_to_module=None, compound=None, coords_only=False,
         rigid=False, use_parmed=False, **kwargs):
    """Load a file into an mbuild compound.

    Files are read using the MDTraj package unless the `use_parmed` argument is
    specified as True. Please refer to http://mdtraj.org/1.8.0/load_functions.html
    for formats supported by MDTraj and https://parmed.github.io/ParmEd/html/
    readwrite.html for formats supported by ParmEd.

    Parameters
    ----------
    filename : str
        Name of the file from which to load atom and bond information.
    relative_to_module : str, optional, default=None
        Instead of looking in the current working directory, look for the file
        where this module is defined. This is typically used in Compound
        classes that will be instantiated from a different directory
        (such as the Compounds located in mbuild.lib).
    compound : mb.Compound, optional, default=None
        Existing compound to load atom and bond information into.
    coords_only : bool, optional, default=False
        Only load the coordinates into an existing compoint.
    rigid : bool, optional, default=False
        Treat the compound as a rigid body
    use_parmed : bool, optional, default=False
        Use readers from ParmEd instead of MDTraj.
    **kwargs : keyword arguments
        Key word arguments passed to mdTraj for loading.

    Returns
    -------
    compound : mb.Compound

    """
    # Handle mbuild *.py files containing a class that wraps a structure file
    # in its own folder. E.g., you build a system from ~/foo.py and it imports
    # from ~/bar/baz.py where baz.py loads ~/bar/baz.pdb.
    if relative_to_module:
        script_path = os.path.realpath(sys.modules[relative_to_module].__file__)
        file_dir = os.path.dirname(script_path)
        filename = os.path.join(file_dir, filename)

    if compound is None:
        compound = Compound()

    if use_parmed:
        warn("use_parmed set to True.  Bonds may be inferred from inter-particle "
             "distances and standard residue templates!")
        structure = pmd.load_file(filename, structure=True, **kwargs)
        compound.from_parmed(structure, coords_only=coords_only)
    else:
        traj = md.load(filename, **kwargs)
        compound.from_trajectory(traj, frame=-1, coords_only=coords_only)

    if rigid:
        compound.label_rigid_bodies()
    return compound


def clone(existing_compound, clone_of=None, root_container=None):
    """A faster alternative to deepcopying.

    Does not resolve circular dependencies. This should be safe provided
    you never try to add the top of a Compound hierarchy to a
    sub-Compound.def __init__(self, new_origin=None, point_on_x_axis=None,
167                  point_on_xy_plane=None):


    Parameters
    ----------
    existing_compound : mb.Compound
        Existing Compound that will be copied

    Other Parameters
    ----------------
    clone_of : dict, optional
    root_container : mb.Compound, optional

    """
    if clone_of is None:
        clone_of = dict()

    newone = existing_compound._clone(clone_of=clone_of,
                                      root_container=root_container)
    existing_compound._clone_bonds(clone_of=clone_of)
    return newone

    

#def deepen(cmpnd, lookin_for):
#    """rename to sub"""
#    for parti in cmpnd.children:
#        if parti.name is in lookin_for:##### this may be not be kush
#            if parti.n_particles > 0:
#                tempo = cmpnd.center
#                new_positions = _mirror(parti.xyz_with_ports, about= 'x')
#                new_positions.translate_to(tempo)
#                ### is new_positions going to be a compound object? 
#                cmpnd.xyz_with_ports = new_positions
#            else:
#                raise ValueError("The user passed the name of an atom. Even though it "
#                                "exists in the object, it cannot be mirrored because "
#                                "atoms don't have chirality.")  
#        else:            
#            if parti.n_particles > 0:
#                deepen(parti, lookin_for)
#            else:
#                raise ValueError("")
                
class Compound(object):
    """A building block in the mBuild hierarchy.

    Compound is the superclass of all composite building blocks in the mBuild
    hierarchy. That is, all composite building blocks must inherit from
    compound, either directly or indirectly. The design of Compound follows the
    Composite design pattern (Gamma, Erich; Richard Helm; Ralph Johnson; John
    M. Vlissides (1995). Design Patterns Elements of Reusable Object-Oriented
    Software. Addison-Wesley. p. 395. ISBN 0-201-63361-2.), with Compound being
    the composite, and Particle playing the role of the primitive (leaf) part,
    where Particle is in fact simply an alias to the Compound class.

    Compound maintains a list of children (other Compounds contained within),
    and provides a means to tag the children with labels, so that the compounds
    can be easily looked up later. Labels may also point to objects outside the
    Compound's containment hierarchy. Compound has built-in support for copying
    and deepcopying Compound hierarchies, enumerating particles or bonds in the
    hierarchy, proximity based searches, visualization, I/O operations, and a
    number of other convenience methods.

    Parameters
    ----------
    subcompounds : mb.Compound or list of mb.Compound, optional, default=None
        One or more compounds to be added to self.
    name : str, optional, default=self.__class__.__name__
        The type of Compound.
    pos : np.ndarray, shape=(3,), dtype=float, optional, default=[0, 0, 0]
        The position of the Compound in Cartestian space
    charge : float, optional, default=0.0
        Currently not used. Likely removed in next release.
    periodicity : np.ndarray, shape=(3,), dtype=float, optional, default=[0, 0, 0]
        The periodic lengths of the Compound in the x, y and z directions.
        Defaults to zeros which is treated as non-periodic.
    port_particle : bool, optional, default=False
        Whether or not this Compound is part of a Port

    Attributes
    ----------
    bond_graph : mb.BondGraph
        Graph-like object that stores bond information for this Compound
    children : OrderedSet
        Contains all children (other Compounds).
    labels : OrderedDict
        Labels to Compound/Atom mappings. These do not necessarily need not be
        in self.children.
    parent : mb.Compound
        The parent Compound that contains this part. Can be None if this
        compound is the root of the containment hierarchy.
    referrers : set
        Other compounds that reference this part with labels.
    rigid_id : int, default=None
        The ID of the rigid body that this Compound belongs to.  Only Particles
        (the bottom of the containment hierarchy) can have integer values for
        `rigid_id`. Compounds containing rigid particles will always have
        `rigid_id == None`. See also `contains_rigid`.
    boundingbox
    center
    contains_rigid
    max_rigid_id
    n_particles
    n_bonds
    root
    xyz
    xyz_with_ports

    """

    def __init__(self, subcompounds=None, name=None, pos=None, charge=0.0,
                 periodicity=None, port_particle=False):
        super(Compound, self).__init__()

        if name:
            if not isinstance(name, string_types):
                raise ValueError('Compound.name should be a string. You passed '
                                 '{}'.format(name))
            self.name = name
        else:
            self.name = self.__class__.__name__

        # A periodicity of zero in any direction is treated as non-periodic.
        if periodicity is None:
            self._periodicity = np.array([0.0, 0.0, 0.0])
        else:
            self._periodicity = np.asarray(periodicity)

        if pos is not None:
            self._pos = np.asarray(pos, dtype=float)
        else:
            self._pos = np.zeros(3)

        self.parent = None
        self.children = OrderedSet()
        self.labels = OrderedDict()
        self.referrers = set()

        self.bond_graph = None
        self.port_particle = port_particle

        self._rigid_id = None
        self._contains_rigid = False
        self._made_from_lattice = False
        self._check_if_contains_rigid_bodies = False
 
        # self.add() must be called after labels and children are initialized.
        if subcompounds:
            if charge:
                raise MBuildError('Cannot set the charge of a Compound containing '
                                  'subcompounds.')
            self.add(subcompounds)
            self._charge = 0.0
        else:
            self._charge = charge

    def particles(self, include_ports=False):
        """Return all Particles of the Compound.

        Parameters
        ----------
        include_ports : bool, optional, default=False
            Include port particles

        Yields
        -------
        mb.Compound
            The next Particle in the Compound

        """
        if not self.children:
            yield self
        else:
            for particle in self._particles(include_ports):
                yield particle

    def _particles(self, include_ports=False):
        """Return all Particles of the Compound. """
        for child in self.successors():
            if not child.children:
                if include_ports or not child.port_particle:
                    yield child

    def successors(self):
        """Yield Compounds below self in the hierarchy.

        Yields
        -------
        mb.Compound
            The next Particle below self in the hierarchy

        """
        if not self.children:
            return
        for part in self.children:
            # Parts local to the current Compound.
            yield part
            # Parts further down the hierarchy.
            for subpart in part.successors():
                yield subpart

    @property
    def my_label(self):
        """Returns the label of the current compound, as seen by the compound's
        parent.

        The default MBuild labeling convention when building compounds
        is label = "name[{}]".format(ii) where ii is the order (zero indexed)
        which that kind of that compound/particle is added.
        """
        if self.parent:
            for lab in self.parent.labels:
                if isinstance(lab, list):
                    continue
                if self.parent[lab] is self:
                    return lab
                    break
            else:
                raise AttributeError("Developer Error")
                # revisit this error an also should I make this a property?
        else:
            warn ("Object {} is at the top of its hierarchy and thus has no label."
                  " Returning None.".format(self))
            return None

    @property
    def n_particles(self):
        """Return the number of Particles in the Compound.

        Returns
        -------
        int
            The number of Particles in the Compound

        """
        if not self.children:
            return 1
        else:
            return self._n_particles(include_ports=False)

    def _n_particles(self, include_ports=False):
        """Return the number of Particles in the Compound. """
        return sum(1 for _ in self._particles(include_ports))

    def _contains_only_ports(self):
        for part in self.children:
            if not part.port_particle:
                return False
        return True

    def ancestors(self):
        """Generate all ancestors of the Compound recursively.

        Yields
        ------
        mb.Compound
            The next Compound above self in the hierarchy

        """
        if self.parent:
            yield self.parent
            for ancestor in self.parent.ancestors():
                yield ancestor

    @property
    def root(self):
        """The Compound at the top of self's hierarchy.

        Returns
        -------
        mb.Compound
            The Compound at the top of self's hierarchy

        """
        parent = None
        for parent in self.ancestors():
            pass
        if parent is None:
            return self
        return parent

    def particles_by_name(self, name):
        """Return all Particles of the Compound with a specific name

        Parameters
        ----------
        name : str
            Only particles with this name are returned

        Yields
        ------
        mb.Compound
            The next Particle in the Compound with the user-specified name

        """
        ## revisit, what if the particle is not in compound
        for particle in self.particles():
            if particle.name == name:
                yield particle

    def find_particles_in_path(self, within_path):
        """"
        Yields all particles that exist within the hierarchal pathway description provided
         in the parameter 'within_path'.

        :param within_path: accepts list, tuple, or mb.particle
            If a mb.particle is provided, the function will return within_path.
            The a list/tuple is specified, each element is either a str, mb.Compound,
            list/tuple (containing any combination of strs and mb.Compounds). If
            a mb.Compound is provided the function skips to that index and ignores
            all data beyond that index. For example, given ["a0", "a1", <mb.Compound>, "a3"],
            the function would ignore the value "a3" and skip straight to the mb.Compound.
            See description below for more information on hierarchal pathways.

        :yields
            All particles that match the hierarchal description provided.

        A hierarchal pathway is a list or tuple containing any combination of strings,
        list/tuples, or mb.Compounds. Each element of the list/tuple either describes a series
        of subcompounds (this occurs in the instance where an inner list/tuple is passed), or describes
        one subcompound or type of subcompound (when a string is passed), or even IS a subcompound
        (in the instance where a mb.Compound object is passed). Strings correspond
        to either the names or labels of subcompounds, and list/tuples hold multiple strings that
        correspond to names and/or labels. They are used when the user wishes to describe multiple
        pathways, for example, path = [..., ["subcompound[1]", "subcompound[4]"], ...].
        This example demonstrates that the user can describe some but not all of the pathways
        that have the name "subcompound". The order of the elements in the outer list/tuple correspond
        to their position in the hierarchal pathway, where the first index is the lowest level and the
        last is the highest specified. The number of subcompounds the user can specify is unlimited so
        long as each subcompound specified lies within the hierarchy of the list/tuple element that
        follows. In the context of this function, the first index must be a mb.Particle.

        The idea of a pathway is similar to how one sorts through directories on a computer,
        i.e. "C:/user/username/documents" BUT since MBuild uses hierarchal pathways from lowest
        to highest, the MBuild style of writing it would be "documents/username/user/C:".
        # best hierarchal description

        EX: path =["target",
                          ["SubSubSubCompound[0]",
                           "SubSubSubCompound[3]",
                           "SubSubSubCompound[4]"],
                         "SubSubCompound",
                         "SubCompound[6]"]

        TIP:
        The following in an example of when you would pass an inner list/tuple to within_path:
            If you have a monolayer of Free Fatty Acids, each of len 10 and wish to yield the
            H[0] from every other AlkylMonomer, for the pathway parameter you would pass:
                within_path = ["H[0]",
                            ["AlkylMonomer[{}]".format(ii) for ii in range(start=0, stop=10, step=2)], ...]
                    *This examples assumes default labeling behaviour
        This generator recipe can make selecting your path much easier.
        """
        # revisit bc of the list option. i'm worried it won't return any errors. maybe track which ones
        # it doesnt find. That would probably need to be done in find_subc_in_path
        if not isinstance(within_path, (list,tuple)):
            if not isinstance(within_path, mb.Particle):
                raise TypeError("within_path must be of type list or tuple. "
                                "User passed type: {}. within_path can also "
                                "accept a mb.Particle but in this instance "
                                "find_particles_in_path just returns within_path"
                                ".".format(type(within_path)))
            else:
                if within_path._contains_only_ports():
                    yield within_path
                    return
                else:
                    raise TypeError("User passed a subcompound to the parameter within_path. "
                                    "within_path must be of type list or tuple. "
                                    "within_path can also "
                                    "accept a mb.Particle but in this instance "
                                    "find_particles_in_path just returns within_path.")
        if not within_path:
            raise ValueError("within_path cannot be empty.")
        within_path = list(within_path)
        no_yield = True
        parti = within_path[0]
        if len(within_path) ==1:
            if isinstance(parti,mb.Particle):
                if parti._contains_only_ports():
                    yield parti
                    return
                else:
                     raise TypeError("User passed a subcompound within the parameter within_path. "
                                     "within_path must be of type list or tuple and "
                                     "contain a hierarchal pathway. Although subcompounds "
                                     "are accepted in hierarchal pathways, they cannot be "
                                     "the first index in this instance.")
            elif isinstance(parti, str):
                for parts in self:
                    if parts.name == parti or parts.my_label == parti:
                        if no_yield:
                            no_yield = False
                        yield parts
            elif isinstance(parti, (list, tuple)):
                for parts in self:
                    if parts.name in parti or parts.my_label in parti:
                        if no_yield:
                            no_yield = False
                        yield parts
            else:
                raise TypeError("The object contained in within_path is not of "
                                "an acceptable type. Acceptable types are str, tuple,"
                                "list, mb.Particle (for only the first index), "
                                "and mb.Compound (valid for any index except the first)."
                                " User passed object of type: {}.".format(type(parti)))
        elif isinstance(parti, (list,tuple)):
            for subc in self.find_subcompounds_in_path(pathway=within_path[1:]):
                if subc:
                    for parts in subc:
                        if parts.name in parti or parts.my_label in parti:
                            if no_yield:
                                no_yield = False
                            yield parts
        else:
            for subc in self.find_subcompounds_in_path(pathway= within_path[1:]):
                if subc:
                    for parts in subc:
                        if parts.name == parti or parts.my_label == parti:
                            if no_yield:
                                no_yield = False
                            yield parts
        if no_yield:
            raise ValueError("Particle in path {} not found. Verify that "
                             "this is the correct path.".format(within_path))


    def subcompounds_by_name_or_label(self, looking_for):
        """
        Yields all the subcompounds that exhibit the specified name or label.

        Parameters:

        looking_for: accepts str
            This string will specify the name or label of the particle(s) the user wishes
            to find.

        Editors note:
        Whenever calling this function within a function make sure to add in a method to track
        if anything in looking_for was not found
        """
        if isinstance(looking_for, str):
            for parti in self.children:
                if parti.name == looking_for or parti.my_label == looking_for:
                    if not parti._contains_only_ports():
                        yield parti
                    else:
                        raise ValueError("The user passed {}, the name of an atom/ particle within this "
                                        "object. \nPlease use particles_by_name, find_particles_in_path,"
                                         "or a similar \nmethod instead. subcompounds_by_name_or_label "
                                         "is designed to \nyield subcompounds, not particles.".format(parti.name))
                else:
                    if not parti._contains_only_ports():
                        yield from parti.subcompounds_by_name_or_label(looking_for)
                    else:
                        yield None

        elif isinstance(looking_for, (list, tuple)) and all(looking_for):
            for l in looking_for:
                yield from self.subcompounds_by_name_or_label(looking_for=l)
        else:
            raise TypeError("looking_for must be of type str or a list/tuple of strs."
                            " User passed: {}.".format(type(looking_for)))

    def find_subcompounds_in_path(self, pathway):
        """
        yield all subcompounds that are in the specified hierarchal pathway

        :param pathway: list or tuple containing strings, list/tuples or mb.Compounds
            A hierarchal pathway (see below) to the desired subcompounds.

        :return: yields all particles that match the path description.
                yields None if the particle path specified doesn't exist

        A hierarchal pathway is a list or tuple containing any combination of strings,
        list/tuples, or mb.Compounds. Each element of the list/tuple either describes a series
        of subcompounds (this occurs in the instance where an inner list/tuple is passed), or describes
        one subcompound or type of subcompound (when a string is passed), or even IS a subcompound
        (in the instance where a mb.Compound object is passed). Strings correspond
        to either the names or labels of subcompounds, and list/tuples hold multiple strings that
        correspond to names and/or labels. They are used when the user wishes to describe multiple
        pathways, for example, path = [..., ["subcompound[1]", "subcompound[4]"], ...].
        This example demonstrates that the user can describe some but not all of the pathways
        that have the name "subcompound". The order of the elements in the outer list/tuple correspond
        to their position in the hierarchal pathway, where the first index is the lowest level and the
        last is the highest specified. The number of subcompounds the user can specify is unlimited so
        long as each subcompound specified lies within the hierarchy of the list/tuple element that
        follows. In the context of this function, the first index must correspond to a subcompound, not
        a mb.Particle.

        The idea of a pathway is similar to how one sorts through directories on a computer,
        i.e. "C:/user/username/documents" BUT since MBuild uses hierarchal pathways from lowest
        to highest, the MBuild style of writing it would be "documents/username/user/C:".

        EX: path =["target",
                          ["SubSubSubCompound[0]",
                           "SubSubSubCompound[3]",
                           "SubSubSubCompound[4]"],
                         "SubSubCompound",
                         "SubCompound[6]"]

        TIP:
        The following in an example of when you would pass an inner list/tuple to pathway:
            If you have a monolayer of Free Fatty Acids, each of len 10 and wish to yield
            every other AlkylMonomer, for the pathway parameter you would pass:
                pathway = [["AlkylMonomer[{}]".format(ii) for ii in range(start=0, stop=10, step=2)], ...]
            *This examples assumes default labeling behaviour
        This generator recipe can make selecting your path much easier.
        """

        if not isinstance(pathway, (list, tuple)):
            raise TypeError("Parameter pathway must be of type list or tuple. User"
                            " passed type: {}.".format(type(pathway)))
        if not pathway:
            raise ValueError("Parameter 'pathway' cannot be an empty {}.".format(type(pathway)))
        pathway = list(pathway)
        for n, ii in enumerate(pathway):
            if isinstance(ii, (str, list, tuple)):
                pass
            elif isinstance(ii, mb.Compound):
                if pathway[:n]:
                    yield from ii._which_subc(looking_for=pathway[:n])
                else:
                    yield ii
                break
            else:
                raise TypeError("pathway parameter must be either a list or tuple containing"
                                " only strings, lists, tuples or mb.Compounds. User passed {}"
                                " containing invalid type: "
                                "{} at index {}.".format(type(pathway), type(ii), n))
        else:
            yield from self._which_subc(looking_for=pathway)

    def _which_subc(self, looking_for):
        """
        refer to def find_subcompounds_in_path
        """
        shorten = len(looking_for)-1
        lf = looking_for[-1]
        if len(looking_for) > 1:
            we_ok = False
            if isinstance(lf, str):
                for subp in self.subcompounds_by_name_or_label(looking_for=lf):
                    if subp:
                        we_ok = True
                        yield from subp._which_subc(looking_for=looking_for[:shorten])
            else:
                for l in lf:
                    for subp in self.subcompounds_by_name_or_label(looking_for=l):
                        if subp:
                            we_ok = True
                            yield from subp._which_subc(looking_for=looking_for[:shorten])
            if not we_ok:
                yield None
                #raise ValueError('{} was not found within {}'.format(within, self.name))
        else:
            if isinstance(lf, str):
                yield from self.subcompounds_by_name_or_label(looking_for=lf)
            else:
                for l in lf:
                    yield from self.subcompounds_by_name_or_label(looking_for=l)



    # def _bonds_to_neighbors_recurse(self, neigh, same_compound):
    #     """"""""
    #     # can probably trash this function revisit
    #     if self.parent: ### check this
    #         if self.parent is same_compound
    #             # is is always scary
    #             print('None1, from _bonds')
    #             return None
    #         elif self.parent.name == same_compound.name:
    #             return "twin"
    #         elif self.parent.name not in neigh:
    #             return self.parent._bonds_to_neighbors_recurse(neigh, same_compound)
    #         else:
    #             print(self, ' from _bonds')
    #             return True
    #     else:
    #         print('None2, from _bonds')
    #         return None

    def _desired_twins(self, anny, sees_twin):
        """"""
        if anny.my_label in sees_twin:
            if anny.n_particles != self.n_particles:
                warn("Proceed with caution, although these "
                     "subcompounds ({} and {}) have different"
                     " ID's and have the same name, "
                     "they do not have"
                     " the same number of particles."
                     "".format(self, anny))
            return True
        else:
            return False

    def bonds_to_neighbors(self, sees_twin= True, neigh=None):
        """
        sees_twin: a list of numbers or strs


        :yields: returns the neighboring particles

        recipe:
        if you wish to only yield twins (or specific twins) pass either True (or the
        list/tuple containing the names or labels for the appropriate particles) for
        sees_twin and neigh=[]
        """
        if not isinstance(sees_twin, bool):
            if not isinstance(sees_twin,(list,tuple)):
                raise TypeError("sees_twin must be of type list, tuple, "
                                "or bool. User passed type: "
                                "{}.".format(type(sees_twin)))
            see = []
            for en, ii in enumerate(sees_twin):
                if isinstance(ii, str):
                    see.append(ii)
                elif isinstance(ii, int):
                    see.append(self.name+"[{}]".format(ii))
                else:
                    raise TypeError("If sees_twin is of type list or tuple "
                                    "the contents must be either strings or "
                                    "integers corresponding to the labels of the"
                                    " twins that are treated as neighbors. "
                                    "User passed {} of type: {} at index {}"
                                    ".".format(ii, type(ii), en))
            sees_twin=deepcopy(see)
        if neigh is not None:
            if not isinstance(neigh, (list, tuple)):
                # raise TypeError("When the parameter neigh is specified"
                #                 " it must be of type list or tuple. User "
                #                 "passed type: {}.".format(type(neigh)))
                neigh = (neigh,)
            if not all(isinstance(n, str) for n in neigh):
                raise TypeError("All objects within neigh must be of type str.")
            if self._contains_only_ports():
                yield from self._bonds_to_neighbors_using_particle_with_neighbors(neigh=neigh, sees_twin=sees_twin)
            else:
                yield from self._bonds_to_neighbors_with_neighbors(sees_twin=sees_twin, neigh=neigh)
        else:
            if self._contains_only_ports():
                # raise ValueError("If user passes a particle rather than a compound, "
                #                  "user must also pass the parameter 'neigh'.")
                yield from self._bonds_to_neighbors_using_particle_no_neighbors(sees_twin=sees_twin)
            else:
                yield from self._bonds_to_neighbors_no_neighbors(sees_twin=sees_twin)

    def _bonds_to_neighbors_no_neighbors(self, sees_twin):
        """ see bonds_to_neighbors """
        no_yield = True
        for ii in self:
            for bonded_to in self.root.bond_graph.neighbors(ii):
                print(bonded_to)
                # try to figure out a way that i dont double count them
                # or a way that i dont have to go down to the particle level
                # revisit
                # use edgesiter
                twin = False
                # if neigh:
                #     if not isinstance(neigh, list):
                #         raise TypeError("Parameter 'neigh' must be a list of strings."
                #                          " User passed type: {}.".format(type(neigh)))
                #     if any(not isinstance(y, str) for y in neigh):
                #         raise TypeError("Parameter 'neigh' must be a list of strings.")
                #     #revisit this for typeerrors in the future
                #     if bonded_to._bonds_to_neighbors_recurse(neigh, same_compound= self):
                #         yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}
                #     else:
                #         yield {'neighbor' : None, 'you' : ii, 'twin' : twin}
                for anny in bonded_to.ancestors():
                    if anny is self:
                        # revisit
                        # yield {'neighbor' : None, 'you' : ii, 'twin' : twin}
                        break
                    elif anny.name == self.name:
                        if sees_twin:
                            if not isinstance(sees_twin, bool):
                                twin = self._desired_twins(anny= anny, sees_twin=sees_twin)
                                if twin:
                                    if no_yield:
                                        no_yield = False
                                    if anny.n_particles != self.n_particles:
                                        warn("Proceed with caution. Although these"
                                             " {} and {} are bonded to eachother and "
                                             "have the same name, they have a different"
                                             " number of particles.".format(anny, self))
                                    yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}
                                    break
                            else:
                                twin = True
                                if no_yield:
                                    no_yield = False
                                if anny.n_particles != self.n_particles:
                                    warn("Proceed with caution. Although these"
                                         " {} and {} are bonded to eachother and "
                                         "have the same name, they have a different"
                                         " number of particles.".format(anny, self))
                                yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}
                        break
                else:
                    if no_yield:
                        no_yield = False
                    yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}

        if no_yield:
            raise ValueError("Found no neighbors or indicated twins to {}.".format(self))

    def _bonds_to_neighbors_with_neighbors(self, neigh, sees_twin):
        """ see bonds_to_neighbors """
        no_yield = True
        for ii in self:
            for bonded_to in self.root.bond_graph.neighbors(ii):
                twin=False
                for anny in bonded_to.ancestors():
                #     print("anny")
                #     print(type(anny))
                #     print(anny)
                #     print(anny.name)
                #     print("self")
                #     print(self)
                #     print(self.name)
                    if anny is self:
                        break
                    elif anny.name == self.name:
                        print("we are in")
                        if sees_twin:
                            print("are we in?")
                            if not isinstance(sees_twin, bool):
                                twin = self._desired_twins(anny=anny, sees_twin=sees_twin)
                                if twin:
                                    if no_yield:
                                        no_yield = False
                                    if anny.n_particles != self.n_particles:
                                        warn("Proceed with caution. Although these"
                                             " {} and {} are bonded to eachother and "
                                             "have the same name, they have a different"
                                             " number of particles.".format(anny, self))
                                    yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}
                                    break
                            else:
                                twin=True
                                if no_yield:
                                    no_yield = False
                                if anny.n_particles != self.n_particles:
                                    warn("Proceed with caution. Although these"
                                         " {} and {} are bonded to eachother and "
                                         "have the same name, they have a different"
                                         " number of particles.".format(anny, self))
                                yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}
                        break
                    elif anny.name in neigh or anny.my_label in neigh:
                        if no_yield:
                            no_yield = False
                        yield {'neighbor' : bonded_to, 'you' : ii, 'twin' : twin}
                        break
        if no_yield:
            raise ValueError("Found none of the specified neighbors or twins to {}.".format(self))

    def _bonds_to_neighbors_using_particle_no_neighbors(self, sees_twin):
        """see bonds_to_neighbors"""
        no_yield = True
        for bonded_to in self.root.bond_graph.neighbors(self):
            twin = False
            if bonded_to.name==self.name:
                if sees_twin:
                    if isinstance(sees_twin,bool):
                        twin=True
                        if no_yield:
                            no_yield=False
                        yield {"neighbor":bonded_to, "you":self, "twin":twin}
                    else:
                        twin = self._desired_twins(anny=bonded_to, sees_twin=sees_twin)
                        if twin:
                            if no_yield:
                                no_yield=False
                            yield {"neighbor":bonded_to, "you":self, "twin":twin}
            else:
                if no_yield:
                    no_yield=False
                yield {"neighbor":bonded_to, "you":self, "twin":twin}
        if no_yield:
            raise ValueError("Found no neighbors or indicated twins to {}.".format(self))


    def _bonds_to_neighbors_using_particle_with_neighbors(self, neigh, sees_twin):
        """see bonds_to_neighbors"""
        no_yield = True
        for bonded_to in self.root.bond_graph.neighbors(self):
            twin = False
            if bonded_to.name==self.name:
                if sees_twin:
                    if isinstance(sees_twin,bool):
                        twin=True
                        if no_yield:
                            no_yield=False
                        yield {"neighbor":bonded_to, "you":self, "twin":twin}
                    else:
                        twin = self._desired_twins(anny=bonded_to, sees_twin=sees_twin)
                        if twin:
                            if no_yield:
                                no_yield=False
                            yield {"neighbor":bonded_to, "you":self, "twin":twin}
            elif bonded_to.name in neigh or bonded_to.my_label in neigh:
                if no_yield:
                    no_yield=False
                yield {"neighbor":bonded_to, "you":self, "twin":twin}
            else:
                for anny in bonded_to.ancestors():
                    if anny.name in neigh or anny.my_label in neigh:
                        if no_yield:
                            no_yield=False
                        yield {"neighbor":bonded_to, "you":self, "twin":twin}
                        break
        if no_yield:
            raise ValueError("Found none of the specified neighbors or twins to {}.".format(self))

    def find_bonds(self, particle1path= None, particle2path= None):
        """
        # pull 90% of the docstring in bond_swap and place it here
        :param what_bond:
        :return:
        """
        what_bond = list([particle1path, particle2path])
        if not any(what_bond):
            raise ValueError("Both participants of the bond cannot be specified entirely by falsy values")
        to_yield = [0,0] # this will be returned
        good_partis = [[], []] # the next 4 variables will be updated each iteration (2)
        subcs = [[], []]
        all_None = False
        non_nones = [[],[]]
        # non_nones may be unnecessary

        for n, particlepath in enumerate(what_bond):
            if not particlepath or not any(particlepath):
                all_None = n
                continue
            if isinstance(particlepath, (list, tuple)):
                if not (0 < len(particlepath) < 3):
                    raise ValueError("")
            elif isinstance(particlepath, mb.Particle):
                if particlepath._contains_only_ports():
                    print('roof')
                    non_nones[n].append(0)
                    good_partis[n].append(particlepath)
                    # is this working??
                    continue
                else:
                    particlepath = particlepath.my_label
                    print('unusual case happened')
            else:
                raise TypeError("")
            particlepath = list(particlepath)
            print("participant[0]")
            print(particlepath[0])
            print('')
            print("isinstance(participant[0], mb.Particle)")
            print(isinstance(particlepath[0], mb.Particle))
            if isinstance(particlepath[0], mb.Particle):
                if particlepath[0]._contains_only_ports():
                    print('roof')
                    non_nones[n].append(0)
                    good_partis[n].append(particlepath[0])
                    # is this working??
                    continue
                else:
                    particlepath[0] = particlepath[0].my_label
                    print('unusual case happened')
                # make sure I check this enough..... do I need to check other indices

            parti_to_find = None
            looking_for = None
            for p, piece in enumerate(particlepath):
                if p == 0:
                    if piece:
                        if isinstance(piece, str):
                            parti_to_find = piece
                            non_nones[n].append(p)
                        else:
                            raise TypeError("")
                else:
                    if piece:
                        if isinstance(piece, (list, tuple)):
                            non_nones[n].append(p)
                            looking_for = piece
                        else:
                            raise TypeError("")


            # all_None will only evaulate to True on the second go around
            print("all_None")
            print(all_None)
            print("non_nones")
            print(non_nones)
            print("n")
            print(n)
            if not (all_None is False):
                print("isinstance(all_None, int) is True")
                print(isinstance(all_None, int) is True)
                print(all_None)
                ############## for some reason all_None is recognized as an integer BUT WE OK HERE check elsewhere
                if len(non_nones[n]) == 0:
                    raise ValueError("Both participants of the bond cannot be specified entirely by "
                                     "falsy values")
            elif len(non_nones[n]) == 0:
                all_None = n
                print("i see you, suprisingly")
                continue
            # revisit because this may never be reached
            got_one = False
            if looking_for:
                for subc in self.find_subcompounds_in_path(pathway=looking_for):
                    if subc:
                        if parti_to_find:
                            for good_parti in subc.particles(include_ports=False):
                                # this extracts the particles
                                if parti_to_find == good_parti.name or parti_to_find == good_parti.my_label:
                                    good_partis[n].append(good_parti)
                                    if not got_one:
                                        got_one = True
                        else:
                            if not got_one:
                                got_one = True
                            subcs[n].append(subc)
            else:
                for good_parti in self.particles(include_ports=False):
                    if parti_to_find == good_parti.name or parti_to_find == good_parti.my_label:
                        good_partis[n].append(good_parti)
                        if not got_one:
                            got_one = True
            if not got_one:
                raise MBuildError("")
            # if parti_to_find:
            #     if subcs[n]:
            #         for good_parti in it.chain.from_iterable([[parti for parti in \
            #                                            subber.particles(include_ports=False)] \
            #                                           for subber in subc if subber]):
            #             # this extracts the particles
            #             if parti_to_find == good_parti.name or parti_to_find == good_parti.my_label:
            #                 good_partis[n].append(good_parti)
            #     else:
            #         for good_parti in self.particles(include_ports=False):
            #             if parti_to_find == good_parti.name or parti_to_find == good_parti.my_label:
            #                 good_partis[n].append(good_parti)

        nothing_yielded = True
        print("all_None after looping thru both indices of what_bond")
        print(all_None)
        ## review this all_None section
        if not (all_None is False):
            other_index = (all_None+1)%2
            # if (not what_bond[other_index][0]) and \
            #                         len(what_bond[other_index]) - what_bond[other_index].count(None) != 1:
            if 0 not in non_nones[other_index]:
                raise ValueError("If one of the inner tuples/lists contains only falsy values, "
                                     "the other must only contain 1 non-Falsy value at the first index")
            if not good_partis[other_index]:
                raise ValueError('')
            for gp in good_partis[other_index]:
                nayb = gp.root.bond_graph.neighbors(gp)
                if len(nayb) == 1:
                    if nayb[0].root.bond_graph.has_edge(nayb[0], gp):
                        # make sure I am checking bonding correctly
                        if nothing_yielded:
                            nothing_yielded = False
                        to_yield[other_index] = gp
                        to_yield[all_None] = nayb[0]
                        yield to_yield
            if nothing_yielded:
                raise ValueError("If one of the inner tuples/lists contains only falsy values, then the other "
                                     "must contain directions to a particle that only has one neighbor.")
        else:
            count_it = []
            for enum, gp in enumerate(good_partis):
                if not gp:
                    count_it.append(enum)
            if not count_it:
                print("spot check")
                for gp1, gp2 in product(good_partis[0], good_partis[1]):
                    if gp1.root.bond_graph.has_edge(gp1, gp2):
                        if nothing_yielded:
                            nothing_yielded = False
                        yield list([gp1,gp2])
                if nothing_yielded:
                    raise ValueError("No particles were yielded because the bond specified was not found. The "
                                     "specified particles do not neighbor each other.")
            elif len(count_it) ==2:
                if not (subcs[0] and subcs[1]):
                    raise ValueError("where is the subcompound at")
                for s1, s2 in product(subcs[0], subcs[1]):
                    if not (s1 and s2):
                        continue
                    for bonny in s1.bonds_to_neighbors(neigh=s2):
                        if bonny["neighbor"]:
                            if bonny['you'].root.bond_graph.has_edge(bonny["you"], bonny["neighbor"]):
                                # this check may be superfluous, please revisit
                                if nothing_yielded:
                                    nothing_yielded = False
                                yield list(bonny["you"], bonny["neighbor"])
                    if nothing_yielded:
                        raise ValueError("No particles were yielded because the bond specified was not found. The "
                                         "specified subcompounds do not neighbor each other.")
            else:
                c = count_it[0]
                other_index = (c+1)%2
                if not subcs[c]:
                    raise ValueError("where is the subcompound at")
                for p, s in product(good_partis[other_index], subcs[c]):
                    for bonny in p.bonds_to_neighbors(neigh=s):
                        if bonny["neighbor"]:
                            if bonny['you'].root.bond_graph.has_edge(bonny["you"], bonny["neighbor"]):
                                # this check may be superfluous, please revisit
                                if nothing_yielded:
                                    nothing_yielded = False
                                to_yield[other_index] = bonny["you"]
                                to_yield[c] = bonny["neighbor"]
                                yield to_yield
                    if nothing_yielded:
                        raise ValueError("No particles were yielded because the bond specified was not found. The "
                                         "specified subcompound & particle combination are not neighbors.")

            if nothing_yielded:
                raise MBuildError("Nothing yielded, the specified bond was not found. Unexpected error.")



    def bond_swap(self, bond1, bond2, align= False):
        """
        Useful in synthesis and chirality operations
        Given 4 particles--A1, B1, A2, and B2--bonded in the form
        bond1= A1-B1 and bond2= A2-B2, rearrange the bonds so that
        A1-B1, A2-B2 ==> A1-B2, A2-B1, with dashes (-) indicating bonds.

        bond1: tuple/list of len 2 that contains tuples/lists of len 1,2 or 3.

            bond1 = [A1, B1]
            A1 = [A1particle, A1subcompound, A1within]
            B2 = [B1particle, B1subcompound, B1within]
            bond1 = [[A1particle, A1subcompound, A1within], [B1particle, B1subcompound, B1within]]

            The first inner tuple/list in bond1 corresponds to A1, whereas the second corresponds
            to B1 (similarly for bond2, the first inner tuple/list corresponds to A2 and the second
            to B2). The first index of the inner tuple/list (A1/B1particle) is either of type
            mb.Particle, str, or None, which will be used to indicate which particle will act as A1/B1,
            following the example above. If of type mb.particle, A1/B1 only needs to be of length 1,
            since this will sufficiently indicate the object to treat as A1/B1. If str, A1/B1particle is
            either the name or the label of the particle that behaves as A1/B1. If of len 1 (or if the
            following list/tuple indices are None), A1/B1particle is treated as if it is the only particle
            of that name/label to be bonded to B1/A1particle.


            *****************considering starting from the len 3 side of things and working your way down to len1

            The second index of the inner tuple/list (A1/B1subcompound) is also a
            If None is passed for A1/B1particle and A1/B1 is of length 1,
            this means that the particle indicated for B1/A1 only bonded to one particle, A1/B1. If None of length 2


            ##
            The first inner tuple/list in bond1 corresponds to A1, whereas the second corresponds
            to B1 (similarly for bond2, the first inner tuple/list corresponds to A2 and the second
            to B2).

            A1/B1subcompound:
            A1/B1subcompound is optional and accepts type None, mb.compound, or str. If None,
            either the tuple/list corresponding to A1/B1 must be of length 2 or A1/B1within must also
            be set to None. Otherwise, bond_swap finds the subcompound indicated by A1/B1subcompound
            inside of self. If A1/B1subcompound is type str, it is the name or label of the subcompound
            corresponding to where B1/A1 is bonded to. If A1/B1subcompound is type mb.compound, it is the
            subcompound corresponding to where B1/A1 is bonded to. This approach is useful when there are
            multiple instances of particle's with the same name or label as that which participates in the
            bond the user is searching for.

            A1/B1within:
            A1/B1within is optional and accepts type None, mb.compound, or str. If the list/tuple
            corresponding to A1/B1 is of len 3 and bond1[][2] is not None, bond_swap searches for
            a subcompound within self that matches the specifications passed in A1/B1within. This
            approach is used when there are multiple instances of the subcompound specified
            in A1/B1subcompound, and the user wishes to specify the appropriate one that lies within
            A1/B1within. If A1/B1within is type str, it is either the name or label of the mb.compound
            that will be searched for.

            A1/B1particle:
            A1/B1particle is optional and accepts type mb.particle, str, and None. If type mb.particle,
            A1/B1particle is the particle which participates in the bond to B1/A1particle, and thus bond_swap
            requires no further positional arguments. If type str, A1/B1particle is the name or label of
            the corresponding particle in the specified bond. If A1/B1particle is None and the tuple/list
            corresponding to A1/B1 is of length 1 or the values that follow in the tuple/list are None, this
            indicates that A1/B1 is the only particle B1/A1 is bonded to. Otherwise, if A1/B1particle is None
            and the other values in the corresponding list/tuple are not None, A1/B1 is the only particle that
            B1/A1 is bonded to that has the specifications provided in the the latter indices of the tuple/list.

        bond2: tuple/list of len 2 that contains tuples/lists of len 1,2 or 3.
            See description for bond1.

        align: optional, default False, accepts bool.

        :param parts: tuple or list of size 2. parts contains a combination of tuples/lists
                of size 2, strings, or None, although there cannot be #######.
                Examples:
                parts=("C[4]","N")
                parts=(("C[4]","NitroBenzene"),("N[0]","ProteinA[0]"))
                parts=((None, "NitroBenzene"),("N","ProteinA[0]"))
                parts=("C[4]", ("N" , "ProteinA"))
                The first index ("C[4]") of parts corresponds to the particle that makes up
                the bond and exists within self. The second (("N", "ProteinA")) corresponds
                to the exterior particle that self is bonded to. If parts[0] or parts[1] is
                a string, the string specifies either the label or the name of the particle
                that participates in the bond. Otherwise, if it is a tuple/list the first
                index of that tuple is either the name/label of the particle participating
                in the bond or None. The second index is the subcompound (if it exists) that
                the particle exist within. If there are multiple instances of, for example,
                C-N bonds then in order to ensure the operation occurs on the correct C-N
                bonds it is a good idea to specify their labels and/or memberships.
                Ex: parts=(("C[4]","NitroBenzene"),("N[0]","ProteinA[0]")). This specifies
                that we want to return the bond between the fifth C that lies within
                NitroBenzene and the first N that lies within the first ProteinA.
        :return:
        """
        a1, b1 = self.find_bonds(bond1)
        a2, b2 = self.find_bonds(bond2)
        #remove and make the bonds

    # def _mirror(self, anchor, align_position, which_flip):
    #     """"""
    #     which_flip = 1
    #     if len(align_position)>0:
    #         align_position = normalized_matrix(align_position)
    #         norm1 = np.cross(align_position[0],align_position[1])
    #         if np.linalg.norm(norm1) < .045:
    #             # should i do this or should i use angle i would use .055 rad (3.15 deg) as the threshold revisit
    #             # use test cases to try this out
    #             raise ValueError("The vectors passed used to describe the plane are co-linear, thus"
    #                                  " there are infinitely many possible planes.")
    #         norm1 = unit_vector(norm1)
    #         for n, ii in enumerate(np.eye(3)):
    #             if np.allclose(ii, abs(norm1), atol= 1e-6):
    #                 which_flip = n
    #                 align_position = []
    #                 break
    #         else:
    #             moving_align = deepcopy(align_position[0])
    #     # this is a clunky way to do it but i don't know how to thwart getters and setters
    #     # self.xyz_with_ports[:, which_flip] *= -1
    #     new_xyz = deepcopy(self.xyz_with_ports)
    #     new_xyz[:, which_flip] *= -1
    #     self.xyz_with_ports = new_xyz
    #     moving_anchor = deepcopy(anchor)
    #     moving_anchor[which_flip] *=-1
    #     if len(align_position) >0:
    #         print(align_position)
    #         norm2 = deepcopy(norm1)
    #         norm2[which_flip]*=-1
    #         norm2*=-1
    #         moving_align[which_flip] *= -1
    #         self._align(align_these=list([moving_align, norm2]),
    #                    with_these=list([align_position[0],norm1]),
    #                    anchor_pt=moving_anchor)
    #     self.translate(anchor - moving_anchor)
    def _mirror(self, anchor, align_position, which_flip):
        """"""
        # this is a clunky way to do it but i don't know how to thwart getters and setters
        # self.xyz_with_ports[:, which_flip] *= -1
        new_xyz = deepcopy(self.xyz_with_ports)
        new_xyz[:, which_flip] *= -1
        self.xyz_with_ports = new_xyz
        moving_anchor = deepcopy(anchor)
        moving_anchor[which_flip] *=-1
        if len(align_position) >0:
            moving_align = deepcopy(align_position[0])
            print(align_position)
            norm2 = deepcopy(align_position[1])
            norm2[which_flip]*=-1
            norm2*=-1
            moving_align[which_flip] *= -1
            self._align(align_these=list([moving_align, norm2]),
                       with_these=align_position,
                       anchor_pt=moving_anchor)
        self.translate(anchor - moving_anchor)


    def mirror(self, about_vectors=None, mirror_plane_points=None, anchor_point=None, override=False):
        """
        This function mirrors a compound about a mirror plane, then moves it back to an
        anchor point, a point that has the same coorindates before and after the mirroring
        operation.

        The function defaults to mirroring across the "xz" plane. If no anchor point is
        specified and no mirror_plane_points are specified, the cartesian center (self.center)
        of the particle will be treated as an anchor point. If no anchor point is specified,
        but mirror_plane_points are, then all of the mirror_plane_points will be treated as
        anchor points.

        The user can also pass parameters to specify the plane that will be treated
        as a mirror. Since 2 vectors define a plane, the user inputs information
        that will be converted to vectors. Since n-1 vectors are created when n points
        are specified, the user can either pass 2 vectors to about_vectors, the hierarchal
        pathways (description below) of 3 particles to mirror_plane_points, or 1 vector and 2
        particles.


        :param about_vectors: optional, accepts list-like of length 1 or 2 containing
                            list-likes of length 3
            The inner list-likes are 3D vectors. This/these vector(s) will help define
            the plane that will be treated as the mirror plane.

        :param mirror_plane_points: optional, accepts list/tuple of length 2 or 3
            The elements of the list/tuple are also list/tuples containing the hierarchal
            pathways (below) to the particles that will be used to define part or all of the mirror
            plane. These particles will be treated as anchor points if the anchor_point
            parameter is not defined.

        :param anchor_point: optional, accepts list-like
            The list-like provided to anchor_point must either be a unique hierarchal pathway
            (below) or a 3D cartesian coordinate anchor_point is used to define a point that remains
            in the same position before and after the operation. If no anchor point is
            specified and no mirror_plane_points are specified, the cartesian center
            (self.center) of the particle will be treated as an anchor point. If
            mirror_plane_points are provided and anchor_point is None, the mirror_plane_points
            are treated as anchor_points.

        :param override:
        ######## talk w justin and christoph

        A hierarchal pathway is a list or tuple containing any combination of strings,
        list/tuples, or mb.Compounds. Each element of the list/tuple either describes a series
        of subcompounds (this occurs in the instance where an inner list/tuple is passed), or describes
        one subcompound or type of subcompound (when a string is passed), or even IS a subcompound
        (in the instance where a mb.Compound object is passed). Strings correspond
        to either the names or labels of subcompounds, and list/tuples hold multiple strings that
        correspond to names and/or labels. They are used when the user wishes to describe multiple
        pathways, for example, path = [..., ["subcompound[1]", "subcompound[4]"], ...].
        This example demonstrates that the user can describe some but not all of the pathways
        that have the name "subcompound". The order of the elements in the outer list/tuple correspond
        to their position in the hierarchal pathway, where the first index is the lowest level and the
        last is the highest specified. The number of subcompounds the user can specify is unlimited so
        long as each subcompound specified lies within the hierarchy of the list/tuple element that
        follows. In the context of this function, the first index must be a subcompound, not
        a mb.Particle.

        The idea of a pathway is similar to how one sorts through directories on a computer,
        i.e. "C:/user/username/documents" BUT since MBuild uses hierarchal pathways from lowest
        to highest, the MBuild style of writing it would be "documents/username/user/C:".
        # best hierarchal description

        EX: path =["target",
                          ["SubSubSubCompound[0]",
                           "SubSubSubCompound[3]",
                           "SubSubSubCompound[4]"],
                         "SubSubCompound",
                         "SubCompound[6]"]

        TIP:
        The following in an example of when you would pass an inner list/tuple to looking_for:
            If you have a monolayer of Free Fatty Acids, each of len 10 and wish to yield every other
            AlkylMonomer, for the pathway parameter you would pass:
                looking_for = [["AlkylMonomer[{}]".format(ii) for ii in range(start=0, stop=10, step=2)], ...]
            *This examples assumes default labeling behaviour
        This generator recipe can make selecting your path much easier.
        """

        # revisit the idea of latobj
        # in the future try to limit flops

        alignment_vectors= []
        relative_to = None
        print(type(anchor_point))
        if anchor_point is not None:
            if not isinstance(anchor_point, (tuple, list)):
                if not isinstance(anchor_point, np.ndarray):
                    raise TypeError('anchor_point must be of type list, tuple, or np.ndarray.'
                                    ' User passed type: {}.'.format(type(anchor_point)))
                elif len(anchor_point) !=3:
                    raise ValueError("In the instance where a 3D coorindate is described "
                                     "by anchor_point, the coordinate system must be of "
                                     "len 3. User passed len: {}.".format(len(anchor_point)))
                else:
                    relative_to = anchor_point
            else:
                if all(isinstance(ap, (int,float)) for ap in anchor_point):
                    if len(anchor_point) != 3:
                        raise ValueError("In the instance where a 3D coorindate is described "
                                        "by anchor_point, the coordinate system must be of "
                                        "len 3. User passed len: {}.".format(len(anchor_point)))
                    relative_to = np.array(anchor_point)
                else:
                    path_ = deepcopy(anchor_point)
                    anchor_point = list(self.find_particles_in_path(within_path=anchor_point))
                    if len(anchor_point) > 1:
                        raise MBuildError("This is not a unique anchor point. "
                                          "The hierarchal path {} is invalid.".format(path_))
                    relative_to = anchor_point[0].pos

        if mirror_plane_points is not None:
            if not isinstance(mirror_plane_points, (list, tuple)):
                raise TypeError("mirror_plane_points must be of type list or tuple. "
                                "User passed type: {}.".format(type(mirror_plane_points)))
            if len(mirror_plane_points)==3:
                if about_vectors is not None and len(about_vectors)>0:
                    raise ValueError("Overdefined system. Three mirror_plane_points are"
                                     " defined and about_vectors is not None. 2 vectors best"
                                     " describe a plane. Since n-1 vectors are created when n "
                                     "points are described, the mirror plane is overdefined.")
            elif len(mirror_plane_points) == 2:
                if about_vectors is None or len(about_vectors) == 0:
                    raise ValueError("Underdefined system. 2 vectors best describe a "
                                     "plane. Since n-1 vectors are created when n points are "
                                     "described, when mirror_plane_points describes 2 points "
                                     "and about_vectors is None, the mirror plane is underdefined."
                                     " If the system is 2D, please pass (0,0,1) to about_vectors")
                elif len(about_vectors) != 1:
                    if any(isinstance(av, (list,tuple)) for av in about_vectors):
                        raise ValueError("Overdefined system. 2 vectors best describe a plane. Since n-1"
                                         "\nvectors are created when n points are described, when "
                                         "mirror_plane_points describes 2 points and about_vectors describes"
                                         "\nmore than 1 vector, the mirror plane is overdefined.")
                    else:
                        raise TypeError("Parameter about_vectors contains unacceptable types. \n"
                                        "about_vectors must be a list-like of list-likes.")
            else:
                raise ValueError("mirror_plane_points must be either None or a list/"
                                 "tuple of length 2 or 3. User passed length {}."
                                 "".format(len(mirror_plane_points)))
            point = list(self.find_particles_in_path(within_path=mirror_plane_points[0]))
            if len(point) > 1:
                raise MBuildError("{} is not a unique hierarchal pathway. {} particles matched pathway"
                                 ".".format(mirror_plane_points[0], len(point)))
            if relative_to is not None:
                to_vec = point[0].pos
            else:
                relative_to = point[0].pos
                to_vec = relative_to
            for path in mirror_plane_points[1:]:
                point = list(self.find_particles_in_path(within_path=path))
                if len(point) > 1:
                    raise MBuildError("{} is not a unique hierarchal pathway.".format(path))
                alignment_vectors.append(point[0].pos-to_vec)
        if about_vectors:
            if not isinstance(about_vectors, (list, tuple, np.ndarray)):
                raise TypeError("\nabout_vectors must be a list, tuple, or np.ndarray of length 1 "
                                "or 2 that contains any combination\n"
                                " of lists, tuples, and np.ndarrays. User passed type: {} for about_vectors"
                                ".".format(type(about_vectors)))
            if not (1 <= len(about_vectors) <= 2):
                raise ValueError("about_vectors must be of length 1 or 2. Length of {} was passed"
                                 ".".format(len(about_vectors)))
            for av in about_vectors:
                if not isinstance(av, (np.ndarray, tuple, list)):
                    raise TypeError("about_vectors must a list or tuple of any combination of tuples, lists, and"
                                    " np.ndarrays. User passed type: {}.".format(type(av)))
                av = np.array(av)
                if len(av) != 3:
                    raise ValueError("The inner list-likes of about_vectors are of incorrect length. Expected "
                                     "length 3, recieved length {}.".format(len(av)))
                alignment_vectors.append(av)
        if relative_to is None:
            relative_to = self.center
        which_flip = 1
        if alignment_vectors:
            l = len(alignment_vectors)
            # these error messages should never be reached
            if l ==1:
                raise ValueError("The system is underdefined in that it only has 1 vector to describe the "
                                 "plane which the compound will be mirrored across. Planes are best described"
                                 " by 2 vectors. If the compound is 2D please also pass (0,0,1) as an "
                                 "alignment_vector.")
            elif l != 2:
                raise ValueError("The system is overdefined in that has too many vectors that describe the"
                                 " plane it will be mirrored about. Planes are best defined by 2 vectors, "
                                 "user passed arguments which resulted in {} vectors.".format(l))
            alignment_vectors = normalized_matrix(alignment_vectors)
            norm1 = np.cross(alignment_vectors[0],alignment_vectors[1])
            if np.linalg.norm(norm1) < .045:
                # should i do this or should i use angle i would use .055 rad (3.15 deg) as the threshold revisit
                # use test cases to try this out
                raise ValueError("The vectors passed used to describe the plane are co-linear, thus"
                                     " there are infinitely many possible planes.")
            norm1 = unit_vector(norm1)
            for n, ii in enumerate(np.eye(3)):
                if np.allclose(ii, abs(norm1), atol= 5e-4):
                    which_flip = n
                    alignment_vectors = []
                    break
            else:
                alignment_vectors[1] = norm1
        self._mirror(anchor = relative_to, align_position = alignment_vectors, which_flip=which_flip)

    def mirror_child_chirality(self, looking_for,
                               mirror_plane_points=None,
                               keep_orientation_along_vector= None,
                               keep_orientation_with_neighbor= None,
                               anchor_point= None,
                               only_these_twins=True):
        """
        # make sure that user is passing the path of things that lie within the specified subc
        # for things like anchor point and mirror_plane_points

        # copy+ paste a lot from regular def mirror

        tip:
        if user wishes to specify an orientation of a child with respect to another particle,
        find the coordinates of that particle using find_particles_in_path and also the coordinates of the
        subcompound's anchor point (default is .center) you're searching for and make a vector between the two and pass
         them as vectors to the keep_orientation_along_vector parameter


        recipe:
        modifications of this structure will allow the user to specify a way to orient the sought out subcompound
        with a series of particles that are not directly connected to the sought out subcompound
        for parti, location, ap in zip(list_of_partis_to_find, list_of_hierarchal_locations, list_of_anchor_points):
            coors = find_particles_in_path(parti+location)
            subc.mirror_child_chirality(*args, keep_orientation_along_vector= [coors-ap])

        keep_orientation_with_neighbor: dict, optional
                        The keys are strings. Valid strings are those listed in the looking_for parameter.
                        If the key is not in looking_for, an error will be raised. The value pairs are a list
                        of strings of neighboring
                        to a list of str values
                        rename it to keep_bonding_position_with_neighbor

                        This parameter can be used in conjuction with other parameters so that the user can properly
                        align the desired subcompound.
        keep_orientation_along_vector: a dict of str keys to a list of list-like values
        within: accept string, optional
                This parameter is used when there are multiple instances of a subcompound specified in looking_for
                but they have different parents. If the user only wishes to modify the instances within a certain
                parent, the user passes the strings of the parents where mirroring is desired
        """
        # look into support for passing a 3d coordinate to either anchor_point or
        # mirror_plane_points

        # expand the neighbor search so that it only occurs once. do this my making a dictionary of all the bonds
        # you are searching for!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # i am worried about the case where we mirror lattice objects. please look into this. revisit

        # if self._made_from_lattice and not override:
        #     warn("This compound was made from a lattice. If you wish to "
        #          "proceed to change the chirality of its children, pass "
        #          "override= True when calling Compound.mirror(). The "
        #          "object has not been altered by this call.")
        #     # this below error message was used in the precursor, before mirror split in 2
        #                 #         warn('This compound was made from a lattice. It is recommended'
        #                 # " that you use the corresponding lattice object's .mirror()"
        #                 # " method, Ex: some_lattice.mirror(some_compound, 'xy')."
        #                 # ' If you wish to mirror each molecule making up the lattice, '
        #                 # 'pass child_chirality as True and specify the name (str) of '
        #                 # "the object(s) you wish to mirror in a list to the looking_for "
        #                 # "parameter. This call has not changed the object. "
        #                 # 'If you wish to proceed with using Compound.mirror(), '
        #                 # 'include the optional parameter override= True when calling'
        #                 # ' Compound.mirror().')
        #     return

        not_found = True
        if anchor_point is not None:
            if not isinstance(anchor_point, (tuple, list)):
                raise TypeError('anchor_point must be of type list or tuple.'
                                ' User passed type: {}.'.format(type(anchor_point)))
            if not all(isinstance(ap, str) for ap in anchor_point):
                raise TypeError("anchor_point must be a list/tuple of strings that "
                                 "serve as a hierarchal pathway to a unique particle within"
                                 " the subcompound specified in the parameter looking_for.")

        if mirror_plane_points is not None:
            if not isinstance(mirror_plane_points, (list, tuple)):
                raise TypeError("mirror_plane_points must be of type list or tuple. "
                                "User passed type: {}.".format(type(mirror_plane_points)))
            if 2 > len(mirror_plane_points):
                if not keep_orientation_with_neighbor:
                    raise ValueError("mirror_plane_points must be either None or a list/"
                                     "tuple of length between 1 and 3, inclusive. User passed length 1. "
                                     "If user passes length 1, user must also pass arguments "
                                     "to keep_orientation_with_neighbor to describe at least "
                                     "one more point that will be used to create the vectors "
                                     "to make the mirror plane.")
            elif 3 < len(mirror_plane_points):
                raise ValueError("mirror_plane_points must be either None or a list/"
                                 "tuple of length between 1 and 3, inclusive. User passed length {}."
                                 "".format(len(mirror_plane_points)))

        if keep_orientation_with_neighbor is not None:
            if not isinstance(only_these_twins, bool):
                if not isinstance(only_these_twins,(list,tuple)):
                    raise TypeError("only_these_twins must be of type list, tuple, "
                                    "or bool. User passed type: "
                                    "{}.".format(type(only_these_twins)))
                see = []
                for en, ii in enumerate(only_these_twins):
                    if isinstance(ii, str):
                        see.append(ii)
                    elif isinstance(ii, int):
                        see.append(self.name+"[{}]".format(ii))
                    else:
                        raise TypeError("If only_these_twins is of type list or tuple "
                                        "the contents must be either strings or "
                                        "integers corresponding to the labels of the"
                                        " twins that are treated as neighbors. "
                                        "User passed {} of type: {} at index {}"
                                        ".".format(ii, type(ii), en))
                only_these_twins=deepcopy(see)
            if not isinstance(keep_orientation_with_neighbor, (list, tuple)):
                raise TypeError("When the parameter keep_orientation_with_neighbor is specified"
                                " it must be of type list or tuple. User "
                                "passed type: {}.".format(type(keep_orientation_with_neighbor)))
            if not all(isinstance(n, str) for n in keep_orientation_with_neighbor):
                raise TypeError("All objects within keep_orientation_with_neighbor must be of type str.")


        if keep_orientation_along_vector is not None:
            if not isinstance(keep_orientation_along_vector, (list, tuple, np.ndarray)):
                raise TypeError("Parameter keep_orientation_along_vector must be list-like. "
                                "User passed type: {}.".format(type(keep_orientation_along_vector)))
            if not (1<=len(keep_orientation_along_vector)<=2):
                raise ValueError("Parameter keep_orientation_along_vector must be of length 1 or 2."
                                 " User passed length: {}.".format(len(keep_orientation_along_vector)))
        else:
            keep_orientation_along_vector=[]


        for subc in self.find_subcompounds_in_path(pathway=looking_for):
            if not subc:
                continue
            # if subc._made_from_lattice and not lat_obj:
            #     raise ValueError("{} was made from a lattice but it was not passed as a key in lat_obj."
            #                      " Please pass a dict with key {} and value pair as the corresponding "
            #                      "lattice object, OR None. If value pair is None, this will treat {}"
            #                      " as a compound. The risk associated with doing this is that it "
            #                      "interferes with the lattice vectors and thus all Lattice methods that "
            #                      "operate on those (e.g. redo, undo, populate, etc)"
            #                      ".".format(subc.name, subc.name, subc.name))
            # this error message is REEEEAAALly wrong
            if not_found:
                not_found = False
            relative_to = None
            if anchor_point is not None:
                path_ = list(self.find_particles_in_path(within_path=anchor_point))
                if len(path_) > 1:
                    raise MBuildError("This is not a unique anchor point. "
                                      "The hierarchal path {} is invalid.".format(anchor_point))
                relative_to = path__[0].pos
            keeper = []
            keeper.extend(keep_orientation_along_vector)
            mirror_points = []
            if mirror_plane_points is not None:
                if len(keeper)==2:
                    raise MBuildError("Overdefined system. mirror_child_chirality requires data "
                                      "that will sufficiently describe a mirror plane. This is "
                                      "done by 2 vectors. Since the user has already supplied 2"
                                      " vectors in keep_orientation_along_vector, this system is "
                                      "overdefined.")
                for path in mirror_plane_points:
                    point = list(self.find_particles_in_path(within_path=path))
                    if len(point) > 1:
                        raise MBuildError("{} is not a unique hierarchal pathway.".format(path))
                    mirror_points.append(point[0].pos)

            bords=[]
            nays=[]
            if keep_orientation_with_neighbor is not None:
                if len(keeper)==2 or (len(keeper) + len(mirror_points))>=3:
                    raise MBuildError("Overdefined system. mirror_child_chirality requires data "
                                      "that will sufficiently describe a mirror plane. This is "
                                      "done by 2 vectors. Since the user has already supplied "
                                      "information to describe 2 or more vectors in "
                                      "keep_orientation_along_vector and mirror_plane_points")
                if isinstance(keep_orientation_with_neighbor, bool):
                    for n in tuple(subc._bonds_to_neighbors_no_neighbors(sees_twin= only_these_twins)):
                        bords.append(n["you"].pos)
                        nays.append(n['neighbor'].pos)
                    if len(bords)>3:
                        raise MBuildError("Overdefined system. The subcompound "
                                          "{} has too many neighbors ({}), since "
                                          "a plane is best described by 3 points"
                                          ".".format(subc, len(bords)))
                else:
                    for n in tuple(subc._bonds_to_neighbors_with_neighbors(sees_twin= only_these_twins,
                                                           neigh= keep_orientation_with_neighbor)):
                        bords.append(n["you"].pos)
                        nays.append(n['neighbor'].pos)
                    if len(bords)>3:
                        raise MBuildError("Overdefined system. The subcompound "
                                          "{} has too many neighbors ({}), since "
                                          "a plane is best described by 3 points"
                                          ".".format(subc, len(bords)))
                # for n in tuple(subc.bonds_to_neighbors(sees_twin= only_these_twins,
                #                                        neigh= keep_orientation_with_neighbor)):
                #     # if not n['neighbor']:
                #     #     continue
                #     # revisit commenting this out
                #     # you should append the neighbor to the hidden_helpers and the ones inside of the
                #         # subc to keeper
                #     bords.append(n["you"].pos)
                #     nays.append(n['neighbor'].pos)
                # if len(bords)>3:
                #     raise MBuildError("Overdefined system. The subcompound "
                #                       "{} has too many neighbors ({}), since "
                #                       "a plane is best described by 3 points"
                #                       ".".format(subc, len(bords)))
                #### probably nix everything below
                # if len(bords) ==1:
                #     if potential_relative_to is not None:
                #         keeper.append(potential_relative_to-bords[0])
                #     elif len(keeper) == 1:
                #         if relative_to:
                #             keeper.append(relative_to-bords[0])
                #         else:
                #             raise MBuildError("Underdefined system.")
                # elif len(bords) == 2:
                #     if len(keeper) == 1:
                #         keeper.append(bords[0]-bords[1])
                #     elif relative_to############################# start here tomorrow
                # elif len(bords) == 3:
                #     pass
                # else:
                #     raise MBuildError()
                # if anchor_point or (len(nays) == 1):
                #     keeper.extend([relative_to - nay for nay in nays])
                # else:
                #     relative_to = np.mean(nays, axis = 0)
                #     for vec in nays[1:]:
                #         keeper.append(nays[0]-vec)
                #     # consider adding this in to ensure these have neighbors: has_neighbor = True
                #     # revisit check overdefined case
            which_flip=1
            s = len(bords) + len(mirror_points)
            check_please = True
            if s == 1:
                raise MBuildError("The vectors used to describe the orientation of the mirror plane.......")
            if len(keeper) == 0:
                if s > 3:
                    raise MBuildError("Overdefined system.")
                elif s < 3:
                    raise MBuildError("Underdefined system")
                elif len(bords) == 0:
                    keeper.extend([mirror_points[0]-mirror_points[1], mirror_points[0]-mirror_points[2]])
                else:
                    for ii in product(range(2), repeat=len(nays)):
                        p = list(compress(bords, map(lambda x: x^1, ii)))+list(compress(nays, ii))+mirror_points
                        k = [p[0] - x for x in p[1:]]
                        k = normalized_matrix(k)
                        norm1 = np.cross(k[0],k[1])
                        if np.linalg.norm(norm1) < .045:
                            continue
                        norm1 = unit_vector(norm1)
                        for n, jj in enumerate(np.eye(3)):
                            if np.allclose(jj, abs(norm1), atol= 5e-4):
                                which_flip = n
                                k = []
                                break
                        else:
                            k[1] = norm1
                        break
                    else:
                        # revisit improve this error message
                        raise MBuildError("The vectors passed used to describe the plane are co-linear, thus"
                                          " there are infinitely many possible planes.")
                    keeper = deepcopy(k)
                    check_please=False

            elif len(keeper) ==1:
                if s > 2:
                    raise MBuildError("Overdefined system.")
                elif s < 2:
                    raise MBuildError("Underdefined system")
                elif len(bords) ==0:
                    keeper.append(mirror_points[0]-mirror_points[1])
                else:
                    for ii in product(range(2), repeat=len(nays)):
                        p = list(compress(bords, map(lambda x: x^1, ii)))+list(compress(nays, ii))+mirror_points
                        k = p[0] - p[1]
                        k.extend(keeper)
                        k = normalized_matrix(k)
                        norm1 = np.cross(k[0],k[1])
                        if np.linalg.norm(norm1) < .045:
                            continue
                        norm1 = unit_vector(norm1)
                        for n, jj in enumerate(np.eye(3)):
                            if np.allclose(jj, abs(norm1), atol= 5e-4):
                                which_flip = n
                                k = []
                                break
                        else:
                            k[1] = norm1
                        break
                    else:
                        # revisit improve this error message
                        raise MBuildError("The vectors passed used to describe the plane are co-linear, thus"
                                          " there are infinitely many possible planes.")
                    keeper = deepcopy(k)
                    check_please=False
            elif s != 0:
                raise MBuildError("Overdefined system.")

            if len(keeper) !=2 and which_flip==1:
                raise MBuildError("mirror_child_chirality mirrors a subcompound across a mirror plane. "
                                  "2 vectors best describe a plane but the user has supplied "
                                  "information to describe {} vectors. The compound has not been "
                                  "modified. Please pass information for 2 vectors not {}. "
                                  "Refer to docstring for more information on describing mirror planes"
                                  ".".format(len(keeper), len(keeper)))
            elif check_please:
                keeper = normalized_matrix(keeper)
                norm1 = np.cross(keeper[0],keeper[1])
                if np.linalg.norm(norm1) < .045:
                    # revisit improve this error message
                    raise MBuildError("The vectors passed used to describe the plane are co-linear, thus"
                                      " there are infinitely many possible planes.")
                norm1 = unit_vector(norm1)
                for n, ii in enumerate(np.eye(3)):
                    if np.allclose(ii, abs(norm1), atol= 5e-4):
                        which_flip = n
                        keeper = []
                        break
                else:
                    keeper[1] = norm1
            if relative_to is None:
                if len(bords) > 0:
                    relative_to = bords[0]
                elif len(mirror_points)>0:
                    relative_to = mirror_points[0]
                else:
                    relative_to = subc.center
            subc._mirror(align_position=keeper, anchor=relative_to, which_flip=which_flip)

        if not_found:
            raise ValueError("{} was not found in {}.".format(looking_for[0], self.name))


    # def subcompounds_by_name(self, looking_for):
    #     """Whenever calling this function within a function make sure to add in a method to track
    #     if anything in looking_for was not found"""
    #     if not isinstance(looking_for, list):
    #         raise TypeError("looking_for must be a list of str elements."
    #                         " User passed: {}.".format(type(looking_for)))
    #     for uu in looking_for:
    #         if not isinstance(uu, str):
    #             raise ValueError("looking_for must be a list of str elements.")
    #     for parti in self.children:
    #         #print(parti)
    #         if parti.name in looking_for:
    #             if parti.n_particles > 1:
    #                 yield parti
    #             else:
    #                 raise ValueError("The user passed {}, the name of an atom/ particle within this "
    #                                 "object. Please use the particles_by_name method"
    #                                 " instead.".format(parti.name))
    #         else:
    #             if parti.n_particles > 1:
    #                 #print('1deeper')
    #                 # print('looking_for')
    #                 # print(looking_for)
    #                 yield from parti.subcompounds_by_name(looking_for)
    #             else:
    #                 yield None
    #                 #print("too short")
    #     #print('exit')
    #
    # def which_subc(self, looking_for, within):
    #     """
    #
    #     :param looking_for:
    #     :param within:
    #     :param missing_parent:
    #     :return:
    #     """
    #     if within:
    #         we_ok = False
    #         for subp in self.subcompounds_by_name(looking_for=within):
    #             if subp:
    #                 we_ok = True
    #                 yield from subp.subcompounds_by_name(looking_for=looking_for)
    #         if not we_ok:
    #             raise ValueError('{} was not found within {}'.format(within, self.name))
    #     else:
    #         yield from self.subcompounds_by_name(looking_for= looking_for)
    #
    #
    # def _bonds_to_neighbors_recurse(self, neigh, match):
    #     """"""""
    #     if self.parent: ### check this
    #         if self.parent.name == match:
    #             print('None1, from _bonds')
    #             yield None
    #         elif self.parent.name not in neigh:
    #             yield from self.parent._bonds_to_neighbors_recurse(neigh, match)
    #         else:
    #             print(self, ' from _bonds')
    #             yield self
    #     else:
    #         print('None2, from _bonds')
    #         yield None
    #
    #
    # def bonds_to_neighbors(self, neigh=None):
    #     """
    #
    #     :yields: returns the neighboring particles
    #     """
    #     for ii in self:
    #         for bonded_to in self.root.bond_graph.neighbors(ii):
    #             # try to figure out a way that i dont double count them
    #             # or a way that i dont have to go down to the particle level
    #             if neigh:
    #                 if not isinstance(neigh, list):
    #                     raise TypeError("Parameter 'neigh' must be a list of strings."
    #                                      " User passed type: {}.".format(type(neigh)))
    #                 if any(not isinstance(y, str) for y in neigh):
    #                     raise TypeError("Parameter 'neigh' must be a list of strings.")
    #                 #revisit this for typeerrors in the future
    #                 p = {'neighbor' : (yield bonded_to._bonds_to_neighbors_recurse(neigh, match= self.name)),
    #                                                'you' : ii}
    #                 print('p')
    #                 print(p)
    #                 yield p
    #                 # p = yield from bonded_to._bonds_to_neighbors_recurse(neigh, match= self.name)
    #                 # print(p)
    #                 # yield (('neighbor', p), ('you', ii))
    #             else:
    #                 if self != bonded_to.parent:
    #                       #make sure to check this revisit
    #                     yield {'neighbor' : bonded_to, 'you' : ii}
    #                 else:
    #                     yield {'neighbor' : None, 'you' : ii}
    #
    # def bond_swap(self, parts):
    #     """"""
    #     pass
    #
    # def mirror(self, about= 'xz', override= False, child_chirality= False, looking_for= [], within= None,
    #            bonded_external= [], make_bonds_with= [], break_bonds_with=[], swap_bonds= {},
    #            keep_orientation_along_vector= {}, keep_orientation_with_neighbor= {}):
    #     """
    #     consider breaking this up into 2 functions
    #
    #     which flip will default to the largest
    #     which flip if
    #     I need a better name for the child chirality flag
    #
    #     what i need to do now is to figure out how to implement the align_vectors method. i think
    #     what i will do is allow either 1 or 2 args, with a flag to indicate if theyre points or
    #     directions. if they are points, make vectors between them and find the orthagonal. upon finding the orthagonal,
    #
    #     keep_orientation_with_neighbor: dict, optional
    #                     The keys are strings. Valid strings are those listed in the looking_for parameter.
    #                     If the key is not in looking_for, an error will be raised. The value pairs are a list
    #                     of strings of neighboring
    #                     to a list of str values
    #
    #                     This parameter can be used in conjuction with other parameters so that the user can properly
    #                     align the desired subcompound.
    #     keep_orientation_along_vector: a dict of str keys to a list of list-like values
    #     within: accept string, optional
    #             This parameter is used when there are multiple instances of a subcompound specified in looking_for
    #             but they have different parents. If the user only wishes to modify the instances within a certain
    #             parent, the user passes the strings of the parents where mirroring is desired
    #     """
    #
    #     #consider adding support for recenter for the non chiral situation
    #
    #     # expand the neighbor search so that it only occurs once. do this my making a dictionary of all the bonds
    #     # you are searching for!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    #     if child_chirality:
    #         danger_dict = {}
    #         if bonded_external:
    #             if len(bonded_external) != len(looking_for):
    #                 raise ValueError("looking_for must be of same length as bonded_external")
    #             else:
    #                 for ii, jj in zip(looking_for, bonded_external):
    #                     if jj:
    #                         danger_dict.setdefault(ii,jj)
    #
    #
    #
    #         if self.made_from_lattice and not override:
    #             warn("This compound was made from a lattice. If you wish to "
    #                  "proceed to change the chirality of its children, pass "
    #                  "override= True when calling Compound.mirror(). The "
    #                  "object has not been altered by this call.")
    #             return
    #         if make_bonds_with:
    #             pass
    #
    #         missing_looking_for = deepcopy(looking_for)
    #         if within:
    #             if not isinstance(within, str):
    #                 raise ValueError("Parameter 'within' must be of type str. User passed: {}".format(type(within)))
    #             within = [within]
    #         for subc in self.which_subc(looking_for=looking_for, within=within):
    #             if not subc:
    #                 continue
    #             if subc.name in missing_looking_for:
    #                 missing_looking_for.remove(subc.name)
    #             if subc.name in danger_dict:
    #                 pass
    #                 #bondswap
    #
    #             keeper = []
    #             if (subc.name in keep_orientation_along_vector) and keep_orientation_along_vector:
    #                 keeper.extend(keep_orientation_along_vector)
    #             if (subc.name in keep_orientation_with_neighbor) and keep_orientation_with_neighbor:
    #                 missing_neighbor = deepcopy(keep_orientation_with_neighbor)
    #                 for n, nn in tuple(subc.bonds_to_neighbors(neigh= keep_orientation_with_neighbor)):
    #                     print('n')
    #                     print(type(n))
    #                     print(n)
    #                     n = dict(n)
    #                     if not n['neighbor']:
    #                         continue
    #                     nay = n['neighbor']
    #                     print('nay')
    #                     print(type(nay))
    #                     print(nay)
    #                     if nay.parent.name in missing_neighbor:
    #                         missing_neighbor.remove(nay.parent.name)
    #                     keeper.append(subc.center - nay.pos)
    #                 if len(missing_neighbor) > 0:
    #                     raise ValueError ('{} does not neighbor {}'.format(missing_neighbor, subc.name))
    #
    #             _mirror(subc, about, recenter= True, keep_position= keeper) #keep_position_with)
    #             if swap_bonds:
    #                 subc.bond_swap(swap_bonds[subc.name])
    #
    #
    #         if len(missing_looking_for) > 0:
    #             raise ValueError("One or more of the following names do not "
    #                              "exist in {}. Missing: {}.".format(self.name, missing_looking_for))
    #
    #     elif keep_orientation_with_neighbor or keep_orientation_along_vector or make_bonds_with or\
    #             break_bonds_with or looking_for or within or swap_bonds:
    #         warn('Overdefined system. When child_chirality is False, .mirror() only accepts arguments:'
    #              ' about, override. No action was taken, the object has not been modified.')
    #     else:
    #         if self.made_from_lattice:
    #             if override:
    #                 _mirror(self, about, recenter= False)
    #             else:
    #                 warn('This compound was made from a lattice. It is recommended'
    #                     " that you use the corresponding lattice object's .mirror()"
    #                     " method, Ex: some_lattice.mirror(some_compound, 'xy')."
    #                     ' If you wish to mirror each molecule making up the lattice, '
    #                     'pass child_chirality as True and specify the name (str) of '
    #                     "the object(s) you wish to mirror in a list to the looking_for "
    #                     "parameter. This call has not changed the object. "
    #                     'If you wish to proceed with using Compound.mirror(), '
    #                     'include the optional parameter override= True when calling'
    #                     ' Compound.mirror().')

    @property
    def charge(self):
        return sum([particle._charge for particle in self.particles()])

    @charge.setter
    def charge(self, value):
        if self._contains_only_ports():
            self._charge = value
        else:
            raise AttributeError("charge is immutable for Compounds that are "
                                 "not at the bottom of the containment hierarchy.")

    @property
    def rigid_id(self):
        return self._rigid_id

    @rigid_id.setter
    def rigid_id(self, value):
        if self._contains_only_ports():
            self._rigid_id = value
            for ancestor in self.ancestors():
                ancestor._check_if_contains_rigid_bodies = True
        else:
            raise AttributeError("rigid_id is immutable for Compounds that are "
                                 "not at the bottom of the containment hierarchy.")

    @property
    def contains_rigid(self):
        """Returns True if the Compound contains rigid bodies

        If the Compound contains any particle with a rigid_id != None
        then contains_rigid will return True. If the Compound has no
        children (i.e. the Compound resides at the bottom of the containment
        hierarchy) then contains_rigid will return False.

        Returns
        -------
        bool
            True if the Compound contains any particle with a rigid_id != None

        Notes
        -----
        The private variable '_check_if_contains_rigid_bodies' is used to help
        cache the status of 'contains_rigid'. If '_check_if_contains_rigid_bodies'
        is False, then the rigid body containment of the Compound has not changed,
        and the particle tree is not traversed, boosting performance.

        """
        if self._check_if_contains_rigid_bodies:
            self._check_if_contains_rigid_bodies = False
            if any(particle.rigid_id is not None for particle in self._particles()):
                self._contains_rigid = True
            else:
                self._contains_rigid = False
        return self._contains_rigid

    @property
    def max_rigid_id(self):
        """Returns the maximum rigid body ID contained in the Compound.

        This is usually used by compound.root to determine the maximum
        rigid_id in the containment hierarchy.

        Returns
        -------
        int or None
            The maximum rigid body ID contained in the Compound. If no
            rigid body IDs are found, None is returned

        """
        try:
            return max([particle.rigid_id for particle in self.particles()
                        if particle.rigid_id])
        except ValueError:
            return

    def rigid_particles(self, rigid_id=None):
        """Generate all particles in rigid bodies.

        If a rigid_id is specified, then this function will only yield particles
        with a matching rigid_id.

        Parameters
        ----------
        rigid_id : int, optional
            Include only particles with this rigid body ID

        Yields
        ------
        mb.Compound
            The next particle with a rigid_id that is not None, or the next
            particle with a matching rigid_id if specified

        """
        for particle in self.particles():
            if rigid_id:
                if particle.rigid_id == rigid_id:
                    yield particle
            else:
                if particle.rigid_id:
                    yield particle

    def label_rigid_bodies(self, discrete_bodies=None, rigid_particles=None):
        """Designate which Compounds should be treated as rigid bodies

        If no arguments are provided, this function will treat the compound
        as a single rigid body by providing all particles in `self` with the
        same rigid_id. If `discrete_bodies` is not None, each instance of
        a Compound with a name found in `discrete_bodies` will be treated as a
        unique rigid body. If `rigid_particles` is not None, only Particles
        (Compounds at the bottom of the containment hierarchy) matching this name
        will be considered part of the rigid body.

        Parameters
        ----------
        discrete_bodies : str or list of str, optional, default=None
            Name(s) of Compound instances to be treated as unique rigid bodies.
            Compound instances matching this (these) name(s) will be provided
            with unique rigid_ids
        rigid_particles : str or list of str, optional, default=None
            Name(s) of Compound instances at the bottom of the containment
            hierarchy (Particles) to be included in rigid bodies. Only Particles
            matching this (these) name(s) will have their rigid_ids altered to
            match the rigid body number.

        Examples
        --------
        Creating a rigid benzene

        >>> import mbuild as mb
        >>> from mbuild.utils.io import get_fn
        >>> benzene = mb.load(get_fn('benzene.mol2'))
        >>> benzene.label_rigid_bodies()

        Creating a semi-rigid benzene, where only the carbons are treated as
        a rigid body

        >>> import mbuild as mb
        >>> from mbuild.utils.io import                                 
        >>> benzene = mb.load(get_fn('benzene.mol2'))
        >>> benzene.label_rigid_bodies(rigid_particles='C')

        Create a box of rigid benzenes, where each benzene has a unique rigid
        body ID.

        >>> import mbuild as mb
        >>> from mbuild.utils.io import get_fn
        >>> benzene = mb.load(get_fn('benzene.mol2'))
        >>> benzene.name = 'Benzene'
        >>> filled = mb.fill_box(benzene,
        ...                      n_compounds=10,
        ...                      box=[0, 0, 0, 4, 4, 4])
        >>> filled.label_rigid_bodies(distinct_bodies='Benzene')

        Create a box of semi-rigid benzenes, where each benzene has a unique
        rigid body ID and only the carbon portion is treated as rigid.

        >>> import mbuild as mb
        >>> from mbuild.utils.io import get_fn
        >>> benzene = mb.load(get_fn('benzene.mol2'))
        >>> benzene.name = 'Benzene'
        >>> filled = mb.fill_box(benzene,
        ...                      n_compounds=10,
        ...                      box=[0, 0, 0, 4, 4, 4])
        >>> filled.label_rigid_bodies(distinct_bodies='Benzene',
        ...                           rigid_particles='C')

        """
        if discrete_bodies:
            if isinstance(discrete_bodies, string_types):
                discrete_bodies = [discrete_bodies]
        if rigid_particles:
            if isinstance(rigid_particles, string_types):
                rigid_particles = [rigid_particles]

        if self.root.max_rigid_id:
            rigid_id = self.root.max_rigid_id + 1
            warn("{} rigid bodies already exist.  Incrementing 'rigid_id'"
                 "starting from {}.".format(rigid_id, rigid_id))
        else:
            rigid_id = 0

        for successor in self.successors():
            if discrete_bodies and successor.name not in discrete_bodies:
                continue
            for particle in successor.particles():
                if rigid_particles and particle.name not in rigid_particles:
                    continue
                particle.rigid_id = rigid_id
            if discrete_bodies:
                rigid_id += 1

    def unlabel_rigid_bodies(self):
        """Remove all rigid body labels from the Compound """
        self._check_if_contains_rigid_bodies = True
        for child in self.children:
            child._check_if_contains_rigid_bodies = True
        for particle in self.particles():
            particle.rigid_id = None

    def _increment_rigid_ids(self, increment):
        """Increment the rigid_id of all rigid Particles in a Compound

        Adds `increment` to the rigid_id of all Particles in `self` that
        already have an integer rigid_id.
        """
        for particle in self.particles():
            if particle.rigid_id:
                particle.rigid_id += increment

    def _reorder_rigid_ids(self):
        """Reorder rigid body IDs ensuring consecutiveness.

        Primarily used internally to ensure consecutive rigid_ids following
        removal of a Compound.

        """
        max_rigid = self.max_rigid_id
        unique_rigid_ids = sorted(set([p.rigid_id for p in self.rigid_particles()]))
        n_unique_rigid = len(unique_rigid_ids)
        if max_rigid and n_unique_rigid != max_rigid + 1:
            missing_rigid_id = (unique_rigid_ids[-1] * (unique_rigid_ids[-1] + 1))/2 - sum(unique_rigid_ids)
            for successor in self.successors():
                if successor.rigid_id:
                    if successor.rigid_id > missing_rigid_id:
                        successor.rigid_id -= 1
            if self.rigid_id:
                if self.rigid_id > missing_rigid_id:
                    self.rigid_id -= 1

    def add(self, new_child, label=None, containment=True, replace=False,
            inherit_periodicity=True, reset_rigid_ids=True):
        """Add a part to the Compound.

        Note:
            This does not necessarily add the part to self.children but may
            instead be used to add a reference to the part to self.labels. See
            'containment' argument.

        Parameters
        ----------
        new_child : mb.Compound or list-like of mb.Compound
            The object(s) to be added to this Compound.
        label : str, optional
            A descriptive string for the part.
        containment : bool, optional, default=True
            Add the part to self.children.
        replace : bool, optional, default=True
            Replace the label if it already exists.
        inherit_periodicity : bool, optional, default=True
            Replace the periodicity of self with the periodicity of the
            Compound being added
        reset_rigid_ids : bool, optional, default=True
            If the Compound to be added contains rigid bodies, reset the
            rigid_ids such that values remain distinct from rigid_ids
            already present in `self`. Can be set to False if attempting
            to add Compounds to an existing rigid body.

        """
        # Support batch add via lists, tuples and sets.
        if (isinstance(new_child, collections.Iterable) and
                not isinstance(new_child, string_types)):
            for child in new_child:
                self.add(child, reset_rigid_ids=reset_rigid_ids)
            return

        if not isinstance(new_child, Compound):
            raise ValueError('Only objects that inherit from mbuild.Compound '
                             'can be added to Compounds. You tried to add '
                             '"{}".'.format(new_child))

        if new_child.contains_rigid or new_child.rigid_id is not None:
            if self.contains_rigid and reset_rigid_ids:
                new_child._increment_rigid_ids(increment=self.max_rigid_id + 1)
            self._check_if_contains_rigid_bodies = True
        if self.rigid_id:
            self.rigid_id = None

        # Create children and labels on the first add operation
        if self.children is None:
            self.children = OrderedSet()
        if self.labels is None:
            self.labels = OrderedDict()

        if containment:
            if new_child.parent:
                raise MBuildError('Part {} already has a parent: {}'.format(
                    new_child, new_child.parent))
            self.children.add(new_child)
            new_child.parent = self

            if new_child.bond_graph:
                if self.root.bond_graph is None:
                    self.root.bond_graph = new_child.bond_graph
                else:
                    self.root.bond_graph.compose(new_child.bond_graph)

                new_child.bond_graph = None

        # Add new_part to labels. Does not currently support batch add.
        if label is None:
            label = '{0}[$]'.format(new_child.name)

        if label.endswith('[$]'):
            label = label[:-3]
            if label not in self.labels:
                self.labels[label] = []
            label_pattern = label + '[{}]'

            count = len(self.labels[label])
            self.labels[label].append(new_child)
            label = label_pattern.format(count)

        if not replace and label in self.labels:
            raise MBuildError('Label "{0}" already exists in {1}.'.format(
                label, self))
        else:
            self.labels[label] = new_child
        new_child.referrers.add(self)

        if (inherit_periodicity and isinstance(new_child, Compound) and
                new_child.periodicity.any()):
            self.periodicity = new_child.periodicity

    def remove(self, objs_to_remove):
        """Remove children from the Compound.

        Parameters
        ----------
        objs_to_remove : mb.Compound or list of mb.Compound
            The Compound(s) to be removed from self

        """
        if not self.children:
            return

        if not hasattr(objs_to_remove, '__iter__'):
            objs_to_remove = [objs_to_remove]
        objs_to_remove = set(objs_to_remove)

        if len(objs_to_remove) == 0:
            return

        remove_from_here = objs_to_remove.intersection(self.children)
        self.children -= remove_from_here
        yet_to_remove = objs_to_remove - remove_from_here

        for removed in remove_from_here:
            for child in removed.children:
                removed.remove(child)

        for removed_part in remove_from_here:
            if removed_part.rigid_id:
                for ancestor in removed_part.ancestors():
                    ancestor._check_if_contains_rigid_bodies = True
            if self.root.bond_graph and self.root.bond_graph.has_node(removed_part):
                self.root.bond_graph.remove_node(removed_part)
            self._remove_references(removed_part)

        # Remove the part recursively from sub-compounds.
        for child in self.children:
            child.remove(yet_to_remove)
            if child.contains_rigid:
                self.root._reorder_rigid_ids()

    def _remove_references(self, removed_part):
        """Remove labels pointing to this part and vice versa. """
        removed_part.parent = None

        # Remove labels in the hierarchy pointing to this part.
        referrers_to_remove = set()
        for referrer in removed_part.referrers:
            if removed_part not in referrer.ancestors():
                for label, referred_part in list(referrer.labels.items()):
                    if referred_part is removed_part:
                        del referrer.labels[label]
                        referrers_to_remove.add(referrer)
        removed_part.referrers -= referrers_to_remove

        # Remove labels in this part pointing into the hierarchy.
        labels_to_delete = []
        if isinstance(removed_part, Compound):
            for label, part in list(removed_part.labels.items()):
                if not isinstance(part, Compound):
                    for p in part:
                        self._remove_references(p)
                elif removed_part not in part.ancestors():
                    try:
                        part.referrers.discard(removed_part)
                    except KeyError:
                        pass
                    else:
                        labels_to_delete.append(label)
        for label in labels_to_delete:
            removed_part.labels.pop(label, None)

    def referenced_ports(self):
        """Return all Ports referenced by this Compound.

        Returns
        -------
        list of mb.Compound
            A list of all ports referenced by the Compound

        """
        from mbuild.port import Port
        return [port for port in self.labels.values()
                if isinstance(port, Port)]

    def all_ports(self):
        """Return all Ports referenced by this Compound and its successors

        Returns
        -------
        list of mb.Compound
            A list of all Ports referenced by this Compound and its successors

        """
        from mbuild.port import Port
        return [successor for successor in self.successors()
                if isinstance(successor, Port)]

    def available_ports(self):
        """Return all unoccupied Ports referenced by this Compound.

        Returns
        -------
        list of mb.Compound
            A list of all unoccupied ports referenced by the Compound

        """
        from mbuild.port import Port
        return [port for port in self.labels.values()
                if isinstance(port, Port) and not port.used]

    def bonds(self):
        """Return all bonds in the Compound and sub-Compounds.

        Yields
        -------
        tuple of mb.Compound
            The next bond in the Compound

        See Also
        --------
        bond_graph.edges_iter : Iterates over all edges in a BondGraph

        """
        if self.root.bond_graph:
            if self.root == self:
                return self.root.bond_graph.edges_iter()
            else:
                return self.root.bond_graph.subgraph(self.particles()).edges_iter()
        else:
            return iter(())

    @property
    def n_bonds(self):
        """Return the number of bonds in the Compound.

        Returns
        -------
        int
            The number of bonds in the Compound

        """
        return sum(1 for _ in self.bonds())

    def add_bond(self, particle_pair):
        """Add a bond between two Particles.

        Parameters
        ----------
        particle_pair : indexable object, length=2, dtype=mb.Compound
            The pair of Particles to add a bond between

        """
        if self.root.bond_graph is None:
            self.root.bond_graph = BondGraph()

        self.root.bond_graph.add_edge(particle_pair[0], particle_pair[1])

    def generate_bonds(self, name_a, name_b, dmin, dmax):
        """Add Bonds between all pairs of types a/b within [dmin, dmax].

        Parameters
        ----------
        name_a : str
            The name of one of the Particles to be in each bond
        name_b : str
            The name of the other Particle to be in each bond
        dmin : float
            The minimum distance between Particles for considering a bond
        dmax : float
            The maximum distance between Particles for considering a bond

        """
        particle_kdtree = PeriodicCKDTree(data=self.xyz, bounds=self.periodicity)
        particle_array = np.array(list(self.particles()))
        added_bonds = list()
        for p1 in self.particles_by_name(name_a):
            nearest = self.particles_in_range(p1, dmax, max_particles=20,
                                              particle_kdtree=particle_kdtree,
                                              particle_array=particle_array)
            for p2 in nearest:
                if p2 == p1:
                    continue
                bond_tuple = (p1, p2) if id(p1) < id(p2) else (p2, p1)
                if bond_tuple in added_bonds:
                    continue
                min_dist = self.min_periodic_distance(p2.pos, p1.pos)
                if (p2.name == name_b) and (dmin <= min_dist <= dmax):
                    self.add_bond((p1, p2))
                    added_bonds.append(bond_tuple)

    def remove_bond(self, particle_pair):
        """Deletes a bond between a pair of Particles

        Parameters
        ----------
        particle_pair : indexable object, length=2, dtype=mb.Compound
            The pair of Particles to remove the bond between

        """
        if self.root.bond_graph is None or not self.root.bond_graph.has_edge(*particle_pair):
            warn("Bond between {} and {} doesn't exist!".format(*particle_pair))
            return
        self.root.bond_graph.remove_edge(*particle_pair)

    @property
    def pos(self):
        if not self.children:
            return self._pos
        else:
            return self.center

    @pos.setter
    def pos(self, value):
        if not self.children:
            self._pos = value
        else:
            raise MBuildError('Cannot set position on a Compound that has'
                              ' children.')

    @property
    def periodicity(self):
        return self._periodicity

    @periodicity.setter
    def periodicity(self, periods):
        self._periodicity = np.array(periods)

    @property
    def xyz(self):
        """Return all particle coordinates in this compound.

        Returns
        -------
        pos : np.ndarray, shape=(n, 3), dtype=float
            Array with the positions of all particles.
        """
        if not self.children:
            pos = np.expand_dims(self._pos, axis=0)
        else:
            arr = np.fromiter(itertools.chain.from_iterable(
                particle.pos for particle in self.particles()), dtype=float)
            pos = arr.reshape((-1, 3))
        return pos

    @property
    def xyz_with_ports(self):
        """Return all particle coordinates in this compound including ports.

        Returns
        -------
        pos : np.ndarray, shape=(n, 3), dtype=float
            Array with the positions of all particles and ports.

        """
        if not self.children:
            pos = self._pos
        else:
            arr = np.fromiter(itertools.chain.from_iterable(
                particle.pos for particle in self.particles(include_ports=True)), dtype=float)
            pos = arr.reshape((-1, 3))
        return pos

    @xyz.setter
    def xyz(self, arrnx3):
        """Set the positions of the particles in the Compound, excluding the Ports.

        This function does not set the position of the ports.

        Parameters
        ----------
        arrnx3 : np.ndarray, shape=(n,3), dtype=float
            The new particle positions

        """
        if not self.children:
            if not arrnx3.shape[0] == 1:
                raise ValueError('Trying to set position of {} with more than one'
                                 'coordinate: {}'.format(self, arrnx3))
            self.pos = np.squeeze(arrnx3)
        else:
            for atom, coords in zip(self._particles(include_ports=False), arrnx3):
                atom.pos = coords

    @xyz_with_ports.setter
    def xyz_with_ports(self, arrnx3):
        """Set the positions of the particles in the Compound, including the Ports.

        Parameters
        ----------
        arrnx3 : np.ndarray, shape=(n,3), dtype=float
            The new particle positions

        """
        if not self.children:
            if not arrnx3.shape[0] == 1:
                raise ValueError('Trying to set position of {} with more than one'
                                 'coordinate: {}'.format(self, arrnx3))
            self.pos = np.squeeze(arrnx3)
        else:
            for atom, coords in zip(self._particles(include_ports=True), arrnx3):
                atom.pos = coords

    @property
    def center(self):
        """The cartesian center of the Compound based on its Particles.

        Returns
        -------
        np.ndarray, shape=(3,), dtype=float
            The cartesian center of the Compound based on its Particles

        """
        if self.xyz.any():
            return np.mean(self.xyz, axis=0)

    @property
    def boundingbox(self):
        """Compute the bounding box of the compound.

        Returns
        -------
        mb.Box
            The bounding box for this Compound

        """
        xyz = self.xyz
        return Box(mins=xyz.min(axis=0), maxs=xyz.max(axis=0))

    def min_periodic_distance(self, xyz0, xyz1):
        """Vectorized distance calculation considering minimum image.

        Parameters
        ----------
        xyz0 : np.ndarray, shape=(3,), dtype=float
            Coordinates of first point
        xyz1 : np.ndarray, shape=(3,), dtype=float
            Coordinates of second point

        Returns
        -------
        float
            Vectorized distance between the two points following minimum
            image convention

        """
        d = np.abs(xyz0 - xyz1)
        d = np.where(d > 0.5 * self.periodicity, self.periodicity - d, d)
        return np.sqrt((d ** 2).sum(axis=-1))

    def particles_in_range(self, compound, dmax, max_particles=20, particle_kdtree=None,
                           particle_array=None):
        """Find particles within a specified range of another particle.

        Parameters
        ----------
        compound : mb.Compound
            Reference particle to find other particles in range of
        dmax : float
            Maximum distance from 'compound' to look for Particles
        max_particles : int, optional, default=20
            Maximum number of Particles to return
        particle_kdtree : mb.PeriodicCKDTree, optional
            KD-tree for looking up nearest neighbors. If not provided, a KD-
            tree will be generated from all Particles in self
        particle_array : np.ndarray, shape=(n,), dtype=mb.Compound, optional
            Array of possible particles to consider for return. If not
            provided, this defaults to all Particles in self

        Returns
        -------
        np.ndarray, shape=(n,), dtype=mb.Compound
            Particles in range of compound according to user-defined limits

        See Also
        --------
        periodic_kdtree.PerioidicCKDTree : mBuild implementation of kd-trees
        scipy.spatial.ckdtree : Further details on kd-trees

        """
        if particle_kdtree is None:
            particle_kdtree = PeriodicCKDTree(data=self.xyz, bounds=self.periodicity)
        _, idxs = particle_kdtree.query(compound.pos, k=max_particles, distance_upper_bound=dmax)
        idxs = idxs[idxs != self.n_particles]
        if particle_array is None:
            particle_array = np.array(list(self.particles()))
        return particle_array[idxs]

    def visualize(self, show_ports=False):
        """Visualize the Compound using nglview.

        Allows for visualization of a Compound within a Jupyter Notebook.

        Parameters
        ----------
        show_ports : bool, optional, default=False
            Visualize Ports in addition to Particles

        """
        nglview = import_('nglview')
        if run_from_ipython():
            structure = self.to_trajectory(show_ports)
            return nglview.show_mdtraj(structure)
        else:
            raise RuntimeError('Visualization is only supported in Jupyter '
                               'Notebooks.')

    def update_coordinates(self, filename, update_port_locations=True):
        """Update the coordinates of this Compound from a file.

        Parameters
        ----------
        filename : str
            Name of file from which to load coordinates. Supported file types
            are the same as those supported by load()
        update_port_locations : bool, optional, default=True
            Update the locations of Ports so that they are shifted along with
            their anchor particles.  Note: This conserves the location of
            Ports with respect to the anchor Particle, but does not conserve
            the orientation of Ports with respect to the molecule as a whole.

        See Also
        --------
        load : Load coordinates from a file

        """
        if update_port_locations:
            xyz_init = self.xyz
            load(filename, compound=self, coords_only=True)
            self._update_port_locations(xyz_init)
        else:
            load(filename, compound=self, coords_only=True)

    def _update_port_locations(self, initial_coordinates):
        """Adjust port locations after particles have moved

        Compares the locations of Particles between 'self' and an array of
        reference coordinates.  Shifts Ports in accordance with how far anchors
        have been moved.  This conserves the location of Ports with respect to
        their anchor Particles, but does not conserve the orientation of Ports
        with respect to the molecule as a whole.

        Parameters
        ----------
        initial_coordinates : np.ndarray, shape=(n, 3), dtype=float
            Reference coordinates to use for comparing how far anchor Particles
            have shifted.

        """
        particles = list(self.particles())
        for port in self.all_ports():
            if port.anchor:
                idx = particles.index(port.anchor)
                shift = particles[idx].pos - initial_coordinates[idx]
                port.translate(shift)

    def _kick(self):
        """Slightly adjust all coordinates in a Compound

        Provides a slight adjustment to coordinates to kick them out of local
        energy minima.
        """
        xyz_init = self.xyz
        for particle in self.particles():
            particle.pos += (np.random.rand(3,) - 0.5) / 100
        self._update_port_locations(xyz_init)

    def energy_minimization(self, steps=2500, algorithm='cg',
                            forcefield='UFF'):
        """Perform an energy minimization on a Compound

        Utilizes Open Babel (http://openbabel.org/docs/dev/) to perform an
        energy minimization/geometry optimization on a Compound by applying
        a generic force field.

        This function is primarily intended to be used on smaller components,
        with sizes on the order of 10's to 100's of particles, as the energy
        minimization scales poorly with the number of particles.

        Parameters
        ----------
        steps : int, optionl, default=1000
            The number of optimization iterations
        algorithm : str, optional, default='cg'
            The energy minimization algorithm.  Valid options are 'steep',
            'cg', and 'md', corresponding to steepest descent, conjugate
            gradient, and equilibrium molecular dynamics respectively.
        forcefield : str, optional, default='UFF'
            The generic force field to apply to the Compound for minimization.
            Valid options are 'MMFF94', 'MMFF94s', ''UFF', 'GAFF', and 'Ghemical'.
            Please refer to the Open Babel documentation (http://open-babel.
            readthedocs.io/en/latest/Forcefields/Overview.html) when considering
            your choice of force field.

        References
        ----------
        .. [1] O'Boyle, N.M.; Banck, M.; James, C.A.; Morley, C.;
               Vandermeersch, T.; Hutchison, G.R. "Open Babel: An open
               chemical toolbox." (2011) J. Cheminf. 3, 33
        .. [2] Open Babel, version X.X.X http://openbabel.org, (installed
               Month Year)

        If using the 'MMFF94' force field please also cite the following:
        .. [3] T.A. Halgren, "Merck molecular force field. I. Basis, form,
               scope, parameterization, and performance of MMFF94." (1996)
               J. Comput. Chem. 17, 490-519
        .. [4] T.A. Halgren, "Merck molecular force field. II. MMFF94 van der
               Waals and electrostatic parameters for intermolecular
               interactions." (1996) J. Comput. Chem. 17, 520-552
        .. [5] T.A. Halgren, "Merck molecular force field. III. Molecular
               geometries and vibrational frequencies for MMFF94." (1996)
               J. Comput. Chem. 17, 553-586
        .. [6] T.A. Halgren and R.B. Nachbar, "Merck molecular force field.
               IV. Conformational energies and geometries for MMFF94." (1996)
               J. Comput. Chem. 17, 587-615
        .. [7] T.A. Halgren, "Merck molecular force field. V. Extension of
               MMFF94 using experimental data, additional computational data,
               and empirical rules." (1996) J. Comput. Chem. 17, 616-641

        If using the 'MMFF94s' force field please cite the above along with:
        .. [8] T.A. Halgren, "MMFF VI. MMFF94s option for energy minimization
               studies." (1999) J. Comput. Chem. 20, 720-729

        If using the 'UFF' force field please cite the following:
        .. [3] Rappe, A.K., Casewit, C.J., Colwell, K.S., Goddard, W.A. III,
               Skiff, W.M. "UFF, a full periodic table force field for
               molecular mechanics and molecular dynamics simulations." (1992)
               J. Am. Chem. Soc. 114, 10024-10039

        If using the 'GAFF' force field please cite the following:
        .. [3] Wang, J., Wolf, R.M., Caldwell, J.W., Kollman, P.A., Case, D.A.
               "Development and testing of a general AMBER force field" (2004)
               J. Comput. Chem. 25, 1157-1174

        If using the 'Ghemical' force field please cite the following:
        .. [3] T. Hassinen and M. Perakyla, "New energy terms for reduced
               protein models implemented in an off-lattice force field" (2001)
               J. Comput. Chem. 22, 1229-1242
        """
        openbabel = import_('openbabel')

        for particle in self.particles():
            try:
                elem.get_by_symbol(particle.name)
            except KeyError:
                raise MBuildError("Element name {} not recognized. Cannot "
                                  "perform minimization."
                                  "".format(particle.name))

        tmp_dir = tempfile.mkdtemp()
        original = clone(self)
        self._kick()
        self.save(os.path.join(tmp_dir,'un-minimized.mol2'))
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("mol2", "mol2")
        mol = openbabel.OBMol()

        obConversion.ReadFile(mol, os.path.join(tmp_dir, "un-minimized.mol2"))

        ff = openbabel.OBForceField.FindForceField(forcefield)
        if ff is None:
            raise MBuildError("Force field '{}' not supported for energy "
                              "minimization. Valid force fields are 'MMFF94', "
                              "'MMFF94s', 'UFF', 'GAFF', and 'Ghemical'."
                              "".format(forcefield))
        warn("Performing energy minimization using the Open Babel package. Please "
             "refer to the documentation to find the appropriate citations for "
             "Open Babel and the {} force field".format(forcefield))
        ff.Setup(mol)
        if algorithm == 'steep':
            ff.SteepestDescent(steps)
        elif algorithm == 'md':
            ff.MolecularDynamicsTakeNSteps(steps, 300)
        elif algorithm == 'cg':
            ff.ConjugateGradients(steps)
        else:
            raise MBuildError("Invalid minimization algorithm. Valid options "
                              "are 'steep', 'cg', and 'md'.")
        ff.UpdateCoordinates(mol)

        obConversion.WriteFile(mol, os.path.join(tmp_dir, 'minimized.mol2'))
        self.update_coordinates(os.path.join(tmp_dir, 'minimized.mol2'))

    def save(self, filename, show_ports=False, forcefield_name=None,
             forcefield_files=None, box=None, overwrite=False, residues=None,
             references_file=None, combining_rule='lorentz', **kwargs):
        """Save the Compound to a file.

        Parameters
        ----------
        filename : str
            Filesystem path in which to save the trajectory. The extension or
            prefix will be parsed and control the format. Supported
            extensions are: 'hoomdxml', 'gsd', 'gro', 'top', 'lammps', 'lmp'
        show_ports : bool, optional, default=False
            Save ports contained within the compound.
        forcefield_file : str, optional, default=None
            Apply a forcefield to the output file using a forcefield provided
            by the `foyer` package.
        forcefield_name : str, optional, default=None
            Apply a named forcefield to the output file using the `foyer`
            package, e.g. 'oplsaa'. Forcefields listed here:
            https://github.com/mosdef-hub/foyer/tree/master/foyer/forcefields
        box : mb.Box, optional, default=self.boundingbox (with buffer)
            Box information to be written to the output file. If 'None', a
            bounding box is used with 0.25nm buffers at each face to avoid
            overlapping atoms.
        overwrite : bool, optional, default=False
            Overwrite if the filename already exists
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.
        references_file : str, optional, default=None
            Specify a filename to write references for the forcefield that is
            to be applied. References are written in BiBTeX format.
        combining_rule : str, optional, default='lorentz'
            Specify the combining rule for nonbonded interactions. Only relevant
            when the `foyer` package is used to apply a forcefield. Valid
            options are 'lorentz' and 'geometric', specifying Lorentz-Berthelot
            and geometric combining rules respectively.

        Other Parameters
        ----------------
        ref_distance : float, optional, default=1.0
            Normalization factor used when saving to .gsd and .hoomdxml formats
            for converting distance values to reduced units.
        ref_energy : float, optional, default=1.0
            Normalization factor used when saving to .gsd and .hoomdxml formats
            for converting energy values to reduced units.
        ref_mass : float, optional, default=1.0
            Normalization factor used when saving to .gsd and .hoomdxml formats
            for converting mass values to reduced units.

        See Also
        --------
        formats.gsdwrite.write_gsd : Write to GSD format
        formats.hoomdxml.write_hoomdxml : Write to Hoomd XML format
        formats.lammpsdata.write_lammpsdata : Write to LAMMPS data format

        """
        extension = os.path.splitext(filename)[-1]
        if extension == '.xyz':
            traj = self.to_trajectory(show_ports=show_ports)
            traj.save(filename)
            return

        # Savers supported by mbuild.formats
        savers = {'.hoomdxml': write_hoomdxml,
                  '.gsd': write_gsd,
                  '.lammps': write_lammpsdata,
                  '.lmp': write_lammpsdata}

        try:
            saver = savers[extension]
        except KeyError:
            saver = None

        if os.path.exists(filename) and not overwrite:
            raise IOError('{0} exists; not overwriting'.format(filename))

        structure = self.to_parmed(box=box, residues=residues)
        # Apply a force field with foyer if specified
        if forcefield_name or forcefield_files:
            from foyer import Forcefield
            ff = Forcefield(forcefield_files=forcefield_files,
                            name=forcefield_name)
            structure = ff.apply(structure, references_file=references_file)
            structure.combining_rule = combining_rule

        total_charge = sum([atom.charge for atom in structure])
        if round(total_charge, 4) != 0.0:
            warn('System is not charge neutral. Total charge is {}.'
                 ''.format(total_charge))

        # Provide a warning if rigid_ids are not sequential from 0
        if self.contains_rigid:
            unique_rigid_ids = sorted(set([p.rigid_id
                                           for p in self.rigid_particles()]))
            if max(unique_rigid_ids) != len(unique_rigid_ids) - 1:
                warn("Unique rigid body IDs are not sequential starting from zero.")

        if saver:  # mBuild supported saver.
            if extension in ['.gsd', '.hoomdxml']:
                kwargs['rigid_bodies'] = [p.rigid_id for p in self.particles()]
            saver(filename=filename, structure=structure, **kwargs)
        else:  # ParmEd supported saver.
            structure.save(filename, overwrite=overwrite, **kwargs)


    def align_vectors(self, align_these, with_these, anchor_pt = None, lattice_override=False):
        """
        Given 2 sets (align_these and with_these) of 2 vectors, rotate a compound
        so that the vectors align_these point in the direction that with_these do.

        :param align_these: list-like
            The vectors to be aligned. Must represent 3D cartesian coordinates.

        :param with_these: list-like
            The vectors to serve as the end goal for align_these to be aligned with.
            Must represent 3D cartesian coordinates.

        :param anchor_pt: optional, accepts list-like, defaults to self.center
            anchor_pt is used as a way to identify a point in the compound that will remain
            unchanged after alignment. The list-like either contains 3D coordinates or the
            hierarchal pathway to a unique particle that will serve as the anchor point.

        :param lattice_override:
            revisit
        """
        if self._made_from_lattice and not lattice_override:
            warn("This compound was made from a lattice, please use the "
                 "Lattice.rotate(axis_align= True) or "
                 "Lattice.rotate(miller_directions=True) methods."
                 " To proceed use the Compound.align_vectors() method  with "
                 "this compound, pass align_vectors's optional parameter "
                 "lattice_override as True. This compound has not "
                 "been changed.")
            return
        for aligner in list([align_these, with_these]):
            if not isinstance(aligner, (list,tuple)):
                if not isinstance(aligner, np.ndarry):
                    raise TypeError("Parameters align_these and with_these must be a list-like of"
                                    " list-like types.")
                else:
                    aligner = aligner.tolist
            else:
                aligner = list(aligner)
            if len(aligner) !=2:
                raise ValueError("Vector pair {} is not of length 2. Both vectors are required to "
                                 "sufficienly and concisely describe a plane. If you are in"
                                 " the 2D case, please pass (0,0,1) as one of your vectors.".format(aligner))
        ang_current, ang_goal = map(lambda x: angle(x[0], x[1]),
                                    [align_these, with_these])
        if not np.allclose(ang_current, ang_goal, atol= 1e-2):
            raise ValueError("The vectors specified cannot be aligned because the "
                             "angle between the vectors specified in align_these "
                             "is too different from the angle between the vectors "
                             "specified in with_these. Angles were {} and {} degrees, "
                             "respectively.".format(ang_current*180/np.pi,
                                                    ang_goal*180/np.pi))
        align_these, with_these = map(lambda x: normalized_matrix(x), [align_these, with_these])
        if not np.allclose(ang_goal,np.pi/2, atol= 5e-3):
            align_these[1], with_these[1] = map(lambda x: unit_vector(np.cross(x[0], x[1])),
                                                      [align_these, with_these])
            # this ensures that the vector pair will be orthagonal
        if anchor_pt is None:
            anchor_pt = self.center
        else:
            if isinstance(anchor_pt, np.ndarray):
                pass
            elif isinstance(anchor_pt, (tuple, list)):
                if all(isinstance(ap, (int,float)) for ap in anchor_pt):
                    anchor_pt = np.array(anchor_pt)
                else:
                    path = deepcopy(anchor_pt)
                    anchor_pt = list(self.find_particles_in_path(within_path=anchor_pt))
                    # try:
                    #     anchor_pt = list(self.find_particles_in_path(within_path=anchor_pt))
                    # except:
                    #     raise TypeError("The contents, {}, of the {} passed for anchor_pt"
                    #                     " do not contain the appropriate datatypes."
                    #                     " anchor_pt must be either a np.ndarray, list,"
                    #                     " or tuple. If it is a list/tuple, the contents "
                    #                     "must either be 3D coorindates or the hierarchal "
                    #                     "pathway of a unique particle."
                    #                     "".format(anchor_pt, type(anchor_pt)))
                    if len(anchor_pt) > 1:
                        raise MBuildError("This is not a unique anchor point. "
                                          "The hierarchal path {} is invalid."
                                          "".format(path))
                    else:
                        anchor_pt = anchor_pt[0].pos

            else:
                raise TypeError("Parameter anchor_pt must be of type list, tuple, or"
                                " np.ndarray.")

        self._align(align_these=align_these, with_these=with_these,
                    anchor_pt=anchor_pt, lattice_override=lattice_override)


    def _align(self, align_these, with_these, anchor_pt, lattice_override=False):
        """
        This alignment technique assumes that all the input methods have been checked.
        The function align_vectors() checks input and calls upon this to execute the alignment.
        See def align_vectors() for more information.
        """
        current = deepcopy(np.array(align_these))
        goal = np.array(with_these)
        self.translate(-anchor_pt)
        # do error checking
        for ii in range(2):
            if np.allclose(current[ii], goal[ii], atol=1e-3):
                continue
            elif np.allclose(current[ii]*-1, goal[ii], atol= 1e-3):
                self. rotate(theta = np.pi, around= current[(ii+1)%2])
                current[ii]*=-1
                continue
            orthag = np.cross(current[ii], goal[ii])
            theta = abs(angle(current[ii], goal[ii]))
            #current = np.array([Rotation(theta, orthag).apply_to(jj)[0] for jj in current])
            #current = np.array([_rotate(coordinates=jj, theta=theta, around=orthag) for jj in current])
            # current = np.array(list(map(lambda jj : _rotate(coordinates=jj, around=orthag,
            #                                                 theta=theta), current)))
            current = np.array(list(_rotate(coordinates=current, around=orthag, theta=theta)))
            current = normalized_matrix(current)
            self.rotate(theta=theta, around=orthag)
            # compare the end vectors
        self.translate(anchor_pt)



    def translate(self, by):
        """Translate the Compound by a vector

        Parameters
        ----------
        by : np.ndarray, shape=(3,), dtype=float

        """
        new_positions = _translate(self.xyz_with_ports, by)
        self.xyz_with_ports = new_positions
        ### revisit you should add more support for other datatypes

    def translate_to(self, pos):
        # revisit either this one or coordinate_transform._translate_to needs to be deprecated
        """Translate the Compound to a specific position

        Parameters
        ----------
        pos : np.ndarray, shape=3(,), dtype=float

        """
        self.translate(pos - self.center)

    def rotate(self, theta, around, override= False):
        """Rotate Compound around an arbitrary vector.

        Parameters
        ----------
        theta : float
            The angle by which to rotate the Compound, in radians.
        around : np.ndarray, shape=(3,), dtype=float
            The vector about which to rotate the Compound.

        """
        if self._made_from_lattice:
            if override:
                pass
            else:
                warn('This compound was made from a lattice. It is recommended'
                    " that you use the corresponding lattice object's .rotate_lattice()"
                    " method, Ex: some_lattice.rotate_lattice(some_compound, "
                    "new_view= [[1,1,1], np.pi], degrees= False, by_angles= False)."
                    ' This call has not changed the object. '
                    'If you wish to proceed with Compound.rotate_lattice(), '
                    'include the optional parameter override= True when calling'
                    ' Compound.rotate_lattice().')
                return
        new_positions = _rotate(self.xyz_with_ports, theta, around)
        self.xyz_with_ports = new_positions

    def spin(self, theta, around):
        """Rotate Compound in place around an arbitrary vector.

        Parameters
        ----------
        theta : float
            The angle by which to rotate the Compound, in radians.
        around : np.ndarray, shape=(3,), dtype=float
            The axis about which to spin the Compound.

        """
        ###### consider placing a warning message here. revist
        around = np.asarray(around).reshape(3)
        center_pos = self.center
        self.translate(-center_pos)
        self.rotate(theta, around)
        self.translate(center_pos)

    # Interface to Trajectory for reading/writing .pdb and .mol2 files.
    # -----------------------------------------------------------------
    def from_trajectory(self, traj, frame=-1, coords_only=False):
        """Extract atoms and bonds from a md.Trajectory.

        Will create sub-compounds for every chain if there is more than one
        and sub-sub-compounds for every residue.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to load.
        frame : int, optional, default=-1 (last)
            The frame to take coordinates from.
        coords_only : bool, optional, default=False
            Only read coordinate information

        """
        if coords_only:
            if traj.n_atoms != self.n_particles:
                raise ValueError('Number of atoms in {traj} does not match'
                                 ' {self}'.format(**locals()))
            atoms_particles = zip(traj.topology.atoms,
                                  self._particles(include_ports=False))
            for mdtraj_atom, particle in atoms_particles:
                particle.pos = traj.xyz[frame, mdtraj_atom.index]
            return

        atom_mapping = dict()
        for chain in traj.topology.chains:
            if traj.topology.n_chains > 1:
                chain_compound = Compound()
                self.add(chain_compound, 'chain[$]')
            else:
                chain_compound = self
            for res in chain.residues:
                for atom in res.atoms:
                    new_atom = Particle(name=str(atom.name), pos=traj.xyz[frame, atom.index])
                    chain_compound.add(new_atom, label='{0}[$]'.format(atom.name))
                    atom_mapping[atom] = new_atom

        for mdtraj_atom1, mdtraj_atom2 in traj.topology.bonds:
            atom1 = atom_mapping[mdtraj_atom1]
            atom2 = atom_mapping[mdtraj_atom2]
            self.add_bond((atom1, atom2))

        if np.any(traj.unitcell_lengths) and np.any(traj.unitcell_lengths[0]):
            self.periodicity = traj.unitcell_lengths[0]
        else:
            self.periodicity = np.array([0., 0., 0.])

    def to_trajectory(self, show_ports=False, chains=None,
                      residues=None):
        """Convert to an md.Trajectory and flatten the compound.

        Parameters
        ----------
        show_ports : bool, optional, default=False
            Include all port atoms when converting to trajectory.
        chains : mb.Compound or list of mb.Compound
            Chain types to add to the topology
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.

        Returns
        -------
        trajectory : md.Trajectory

        See also
        --------
        _to_topology

        """
        atom_list = [particle for particle in self.particles(show_ports)]

        top = self._to_topology(atom_list, chains, residues)

        # Coordinates.
        xyz = np.ndarray(shape=(1, top.n_atoms, 3), dtype='float')
        for idx, atom in enumerate(atom_list):
            xyz[0, idx] = atom.pos

        # Unitcell information.
        box = self.boundingbox
        unitcell_lengths = np.empty(3)
        for dim, val in enumerate(self.periodicity):
            if val:
                unitcell_lengths[dim] = val
            else:
                unitcell_lengths[dim] = box.lengths[dim]

        return md.Trajectory(xyz, top, unitcell_lengths=unitcell_lengths,
                             unitcell_angles=np.array([90, 90, 90]))

    def _to_topology(self, atom_list, chains=None, residues=None):
        """Create a mdtraj.Topology from a Compound.

        Parameters
        ----------
        atom_list : list of mb.Compound
            Atoms to include in the topology
        chains : mb.Compound or list of mb.Compound
            Chain types to add to the topology
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.

        Returns
        -------
        top : mdtraj.Topology

        See Also
        --------
        mdtraj.Topology : Details on the mdtraj Topology object

        """
        from mdtraj.core.element import get_by_symbol
        from mdtraj.core.topology import Topology

        if isinstance(chains, string_types):
            chains = [chains]
        if isinstance(chains, (list, set)):
            chains = tuple(chains)

        if isinstance(residues, string_types):
            residues = [residues]
        if isinstance(residues, (list, set)):
            residues = tuple(residues)
        top = Topology()
        atom_mapping = {}

        default_chain = top.add_chain()
        default_residue = top.add_residue('RES', default_chain)

        compound_residue_map = dict()
        atom_residue_map = dict()
        compound_chain_map = dict()
        atom_chain_map = dict()

        for atom in atom_list:
            # Chains
            if chains:
                if atom.name in chains:
                    current_chain = top.add_chain()
                    compound_chain_map[atom] = current_chain
                else:
                    for parent in atom.ancestors():
                        if chains and parent.name in chains:
                            if parent not in compound_chain_map:
                                current_chain = top.add_chain()
                                compound_chain_map[parent] = current_chain
                                current_residue = top.add_residue('RES', current_chain)
                            break
                    else:
                        current_chain = default_chain
            else:
                current_chain = default_chain
            atom_chain_map[atom] = current_chain

            # Residues
            if residues:
                if atom.name in residues:
                    current_residue = top.add_residue(atom.name, current_chain)
                    compound_residue_map[atom] = current_residue
                else:
                    for parent in atom.ancestors():
                        if residues and parent.name in residues:
                            if parent not in compound_residue_map:
                                current_residue = top.add_residue(parent.name, current_chain)
                                compound_residue_map[parent] = current_residue
                            break
                    else:
                        current_residue = default_residue
            else:
                if chains:
                    try: # Grab the default residue from the custom chain.
                        current_residue = next(current_chain.residues)
                    except StopIteration: # Add the residue to the current chain
                        current_residue = top.add_residue('RES', current_chain)
                else: # Grab the default chain's default residue
                    current_residue = default_residue
            atom_residue_map[atom] = current_residue

            # Add the actual atoms
            try:
                elem = get_by_symbol(atom.name)
            except KeyError:
                elem = get_by_symbol("VS")
            at = top.add_atom(atom.name, elem, atom_residue_map[atom])
            at.charge = atom.charge
            atom_mapping[atom] = at

        # Remove empty default residues.
        chains_to_remove = [chain for chain in top.chains if chain.n_atoms == 0]
        residues_to_remove = [res for res in top.residues if res.n_atoms == 0]
        for chain in chains_to_remove:
            top._chains.remove(chain)
        for res in residues_to_remove:
            for chain in top.chains:
                try:
                    chain._residues.remove(res)
                except ValueError:  # Already gone.
                    pass

        for atom1, atom2 in self.bonds():
            # Ensure that both atoms are part of the compound. This becomes an
            # issue if you try to convert a sub-compound to a topology which is
            # bonded to a different subcompound.
            if all(a in atom_mapping.keys() for a in [atom1, atom2]):
                top.add_bond(atom_mapping[atom1], atom_mapping[atom2])
        return top

    def from_parmed(self, structure, coords_only=False):
        """Extract atoms and bonds from a pmd.Structure.

        Will create sub-compounds for every chain if there is more than one
        and sub-sub-compounds for every residue.

        Parameters
        ----------
        structure : pmd.Structure
            The structure to load.
        coords_only : bool
            Set preexisting atoms in compound to coordinates given by structure.

        """
        if coords_only:
            if len(structure.atoms) != self.n_particles:
                raise ValueError('Number of atoms in {structure} does not match'
                                 ' {self}'.format(**locals()))
            atoms_particles = zip(structure.atoms,
                                  self._particles(include_ports=False))
            for parmed_atom, particle in atoms_particles:
                particle.pos = np.array([parmed_atom.xx,
                                         parmed_atom.xy,
                                         parmed_atom.xz]) / 10
            return

        atom_mapping = dict()
        chain_id = None
        chains = defaultdict(list)
        for residue in structure.residues:
            chains[residue.chain].append(residue)

        for chain, residues in chains.items():
            if len(chains) > 1:
                chain_compound = Compound()
                self.add(chain_compound, chain_id)
            else:
                chain_compound = self
            for residue in residues:
                for atom in residue.atoms:
                    pos = np.array([atom.xx, atom.xy, atom.xz]) / 10
                    new_atom = Particle(name=str(atom.name), pos=pos)
                    chain_compound.add(new_atom, label='{0}[$]'.format(atom.name))
                    atom_mapping[atom] = new_atom

        for bond in structure.bonds:
            atom1 = atom_mapping[bond.atom1]
            atom2 = atom_mapping[bond.atom2]
            self.add_bond((atom1, atom2))

        if structure.box is not None:
            self.periodicity = structure.box[0:3]
        else:
            self.periodicity = np.array([0., 0., 0.])

    def to_parmed(self, box=None, title='', residues=None):
        """Create a ParmEd Structure from a Compound.

        Parameters
        ----------
        title : str, optional, default=self.name
            Title/name of the ParmEd Structure
        residues : str of list of str
            Labels of residues in the Compound. Residues are assigned by
            checking against Compound.name.

        Returns
        -------
        parmed.structure.Structure
            ParmEd Structure object converted from self

        See Also
        --------
        parmed.structure.Structure : Details on the ParmEd Structure object

        """
        structure = pmd.Structure()
        structure.title = title if title else self.name
        atom_mapping = {}  # For creating bonds below
        guessed_elements = set()

        if isinstance(residues, string_types):
            residues = [residues]
        if isinstance(residues, (list, set)):
            residues = tuple(residues)

        default_residue = pmd.Residue('RES')
        compound_residue_map = dict()
        atom_residue_map = dict()

        for atom in self.particles():
            if residues and atom.name in residues:
                current_residue = pmd.Residue(atom.name)
                atom_residue_map[atom] = current_residue
                compound_residue_map[atom] = current_residue
            elif residues:
                for parent in atom.ancestors():
                    if residues and parent.name in residues:
                        if parent not in compound_residue_map:
                            current_residue = pmd.Residue(parent.name)
                            compound_residue_map[parent] = current_residue
                        atom_residue_map[atom] = current_residue
                        break
                else:  # Did not find specified residues in ancestors.
                    current_residue = default_residue
                    atom_residue_map[atom] = current_residue
            else:
                current_residue = default_residue
                atom_residue_map[atom] = current_residue

            if current_residue not in structure.residues:
                structure.residues.append(current_residue)

            atomic_number = None
            name = ''.join(char for char in atom.name if not char.isdigit())
            try: atomic_number = AtomicNum[atom.name]
            except KeyError:
                element = element_by_name(atom.name)
                if name not in guessed_elements:
                    warn('Guessing that "{}" is element: "{}"'.format(atom, element))
                    guessed_elements.add(name)
            else:
                element = atom.name

            atomic_number = atomic_number or AtomicNum[element]
            mass = Mass[element]
            pmd_atom = pmd.Atom(atomic_number=atomic_number, name=atom.name,
                                mass=mass, charge=atom.charge)
            pmd_atom.xx, pmd_atom.xy, pmd_atom.xz = atom.pos * 10  # Angstroms

            residue = atom_residue_map[atom]
            structure.add_atom(pmd_atom, resname=residue.name,
                               resnum=residue.idx)

            atom_mapping[atom] = pmd_atom

        structure.residues.claim()

        for atom1, atom2 in self.bonds():
            bond = pmd.Bond(atom_mapping[atom1], atom_mapping[atom2])
            structure.bonds.append(bond)
        # pad box with .25nm buffers
        if box is None:
            box = self.boundingbox
            box_vec_max = box.maxs.tolist()
            box_vec_min = box.mins.tolist()
            for dim, val in enumerate(self.periodicity):
                if val:
                    box_vec_max[dim] = val
                    box_vec_min[dim] = 0.0
                if not val:
                    box_vec_max[dim] += 0.25
                    box_vec_min[dim] -= 0.25
            box.mins = np.asarray(box_vec_min)
            box.maxs = np.asarray(box_vec_max)

        box_vector = np.empty(6)
        box_vector[3] = box_vector[4] = box_vector[5] = 90.0
        for dim in range(3):
            box_vector[dim] = box.lengths[dim] * 10
        structure.box = box_vector
        return structure

    def to_intermol(self, molecule_types=None):
        """Create an InterMol system from a Compound.

        Parameters
        ----------
        molecule_types : list or tuple of subclasses of Compound

        Returns
        -------
        intermol_system : intermol.system.System

        """
        from intermol.atom import Atom as InterMolAtom
        from intermol.molecule import Molecule
        from intermol.system import System
        import simtk.unit as u

        if isinstance(molecule_types, list):
            molecule_types = tuple(molecule_types)
        elif molecule_types is None:
            molecule_types = (type(self),)
        intermol_system = System()

        last_molecule_compound = None
        for atom_index, atom in enumerate(self.particles()):
            for parent in atom.ancestors():
                # Don't want inheritance via isinstance().
                if type(parent) in molecule_types:
                    # Check if we have encountered this molecule type before.
                    if parent.name not in intermol_system.molecule_types:
                        self._add_intermol_molecule_type(intermol_system, parent)
                    if parent != last_molecule_compound:
                        last_molecule_compound = parent
                        last_molecule = Molecule(name=parent.name)
                        intermol_system.add_molecule(last_molecule)
                    break
            else:
                # Should never happen if molecule_types only contains type(self)
                raise ValueError('Found an atom {} that is not part of any of '
                                 'the specified molecule types {}'.format(atom, molecule_types))

            # Add the actual intermol atoms.
            intermol_atom = InterMolAtom(atom_index + 1, name=atom.name,
                                         residue_index=1, residue_name='RES')
            intermol_atom.position = atom.pos * u.nanometers
            last_molecule.add_atom(intermol_atom)
        return intermol_system

    @staticmethod
    def _add_intermol_molecule_type(intermol_system, parent):
        """Create a molecule type for the parent and add bonds. """
        from intermol.moleculetype import MoleculeType
        from intermol.forces.bond import Bond as InterMolBond

        molecule_type = MoleculeType(name=parent.name)
        intermol_system.add_molecule_type(molecule_type)

        for index, parent_atom in enumerate(parent.particles()):
            parent_atom.index = index + 1

        for atom1, atom2 in parent.bonds():
            intermol_bond = InterMolBond(atom1.index, atom2.index)
            molecule_type.bonds.add(intermol_bond)

    def __getitem__(self, selection):
        if isinstance(selection, integer_types):
            return list(self.particles())[selection]
        if isinstance(selection, string_types):
            return self.labels.get(selection)

    def __repr__(self):
        descr = list('<')
        descr.append(self.name + ' ')

        if self.children:
            descr.append('{:d} particles, '.format(self.n_particles))
            if any(self.periodicity):
                descr.append('periodicity: {}, '.format(self.periodicity))
            else:
                descr.append('non-periodic, ')
        else:
            descr.append('pos=({: .4f},{: .4f},{: .4f}), '.format(*self.pos))

        descr.append('{:d} bonds, '.format(self.n_bonds))

        descr.append('id: {}>'.format(id(self)))
        return ''.join(descr)

    def _clone(self, clone_of=None, root_container=None):
        """A faster alternative to deepcopying.

        Does not resolve circular dependencies. This should be safe provided
        you never try to add the top of a Compound hierarchy to a
        sub-Compound. Clones compound hierarchy only, not the bonds.
        """
        if root_container is None:
            root_container = self
        if clone_of is None:
            clone_of = dict()

        # If this compound has already been cloned, return that.
        if self in clone_of:
            return clone_of[self]

        # Otherwise we make a new clone.
        cls = self.__class__
        newone = cls.__new__(cls)

        # Remember that we're cloning the new one of self.
        clone_of[self] = newone

        newone.name = deepcopy(self.name)
        newone.periodicity = deepcopy(self.periodicity)
        newone._pos = deepcopy(self._pos)
        newone.port_particle = deepcopy(self.port_particle)
        newone._check_if_contains_rigid_bodies = deepcopy(self._check_if_contains_rigid_bodies)
        newone._contains_rigid = deepcopy(self._contains_rigid)
        newone._rigid_id = deepcopy(self._rigid_id)
        newone._charge = deepcopy(self._charge)
        newone._made_from_lattice = deepcopy(self._made_from_lattice)
        #### look into getter and setter and underscore for the made_from_lattice. revisit
        if hasattr(self, 'index'):
            newone.index = deepcopy(self.index)

        if self.children is None:
            newone.children = None
        else:
            newone.children = OrderedSet()
        # Parent should be None initially.
        newone.parent = None
        newone.labels = OrderedDict()
        newone.referrers = set()
        newone.bond_graph = None

        # Add children to clone.
        if self.children:
            for child in self.children:
                newchild = child._clone(clone_of, root_container)
                newone.children.add(newchild)
                newchild.parent = newone

        # Copy labels, except bonds with atoms outside the hierarchy.
        if self.labels:
            for label, compound in self.labels.items():
                if not isinstance(compound, list):
                    newone.labels[label] = compound._clone(clone_of, root_container)
                    compound.referrers.add(clone_of[compound])
                else:
                    # compound is a list of compounds, so we create an empty
                    # list, and add the clones of the original list elements.
                    newone.labels[label] = []
                    for subpart in compound:
                        newone.labels[label].append(subpart._clone(clone_of, root_container))
                        # Referrers must have been handled already, or the will be handled

        return newone

    def _clone_bonds(self, clone_of=None):
        newone = clone_of[self]
        for c1, c2 in self.bonds():
            try:
                newone.add_bond((clone_of[c1], clone_of[c2]))
            except KeyError:
                raise MBuildError("Cloning failed. Compound contains bonds to "
                                  "Particles outside of its containment hierarchy.")


Particle = Compound

