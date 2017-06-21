
# coding: utf-8

# In[ ]:

from collections import (defaultdict, Iterable, deque)
from copy import deepcopy
import itertools as it
from six import string_types
import numpy as np
from warnings import warn
import mbuild as mb
from mbuild.coordinate_transform import (CoordinateTransform, RotationAroundZ,
                                            RotationAroundY, RotationAroundX, Rotation,
                                            x_axis_transform, y_axis_transform, 
                                             z_axis_transform, angle)

__all__ = ['Lattice']


class Lattice(object):
    """Develop crystal structure from user defined inputs.

    Lattice, the abstract building block of a crystal cell.
    Once defined by the user, the crystal is returned as
    a single Compound that can be either replicated through its class
    methods or through a similar replicate Compound method.

    Lattice is defined through the standard bravais lattices, which have been
    accepted by the International Union of Crystallography.
    A Lattice can be fully described with its lattice vectors and lattice
    spacings. Also, the Lattice can be fully defined by its lattice parameters:
    the lattice spacings and its set of coordinate angles will then
    generate the lattice vectors. Lattice expects a right handed lattice and
    cell edges defined by vectors all originating from the origin in
    Cartesian space.

    Parameters
    ----------
    dimension : int, optional, default=3
        Dimension of the system of interest.
    lattice_vectors : numpy array, shape=(dimension, dimension), optional
                      default=([1,0,0], [0,1,0], [0,0,1])
        Vectors that define edges of unit cell corresponding to dimension.
    lattice_spacings : list-like, shape=(dimension,), optional, default=None
        Length of unit cell edges.
    basis_atoms : dictionary, shape={'id':[nested list of coordinate pairs]}
                    default={'default':[[0., 0., 0.]]
        Location of all basis Compounds in unit cell.
    angles : list-like,  shape=(dimension,), optional, default=None
        Interplanar angles describing unit cell.

    Attributes
    ----------
    dimension : int, optional, default=3
        Dimension of system of interest
    lattice_vectors : numpy array, shape=(dimension, dimension), optional
                      default=([1,0,0], [0,1,0], [0,0,1])
        Vectors that define edges of unit cell corresponding to dimension.
    lattice_spacings : list-like, shape=(dimension,), required, default=None
        Length of unit cell edges.
    basis_atoms : list-like, shape=(['id',[dimension,]], ... ,) optional
                    default={('default',([0,0,0]))}
        Location of all basis Compounds in unit cell.
    angles : list-like, optional, default=None
        Lattice angles to define Bravais Lattice.

    Examples
    --------
    Generating a triclinc lattice for cholesterol.

    >>> import mbuild as mb
    >>> from mbuild.utils.io import get_fn
    >>> # reading in the lattice parameters for crystalline cholesterol
    >>> angle_values = [94.64, 90.67, 96.32]
    >>> spacings = [1.4172, 3.4209, 1.0481]
    >>> basis = {'cholesterol':[[0., 0., 0.]]}
    >>> cholesterol_lattice = mb.Lattice(spacings,
    ...                                  angles=angle_values,
    ...                                  basis_atoms=basis,
    ...                                  dimension=3)

    The lattice based on the bravais lattice parameters of crystalline
    cholesterol was generated.

    Replicating the triclinic unit cell out 3 in x,y,z directions.
    >>> cholesterol_unit = mb.Compound()
    >>> cholesterol_unit = mb.load(get_fn('cholesterol.pdb'))
    >>> # associate basis vector with id 'cholesterol' to cholesterol Compound
    >>> basis_dictionary = {'cholesterol' : cholesterol_unit}
    >>> expanded_cell = cholesterol_lattice.populate(x=3, y=3, z=3,
    ...                              compound_dict=basis_dictionary)

    The unit cell of cholesterol was associated with a Compound that contains
    the connectivity data and spatial arrangements of a cholesterol molecule.
    The unit cell was then expanded out in x,y,z directions and cholesterol
    Compounds were populated.


    Generating BCC CsCl crystal structure
    >>> import mbuild as mb
    >>> chlorine = mb.Compound(name='Cl')
    >>> # angles not needed, when not provided, defaults to 90,90,90
    >>> cesium = mb.Compound(name='Cs')
    >>> spacings = [.4123, .4123, .4123]
    >>> basis = {'Cl' : [[0., 0., 0.]], 'Cs' : [[.5, .5, .5]]}
    >>> cscl_lattice = mb.Lattice(spacings, basis_atoms=basis,
    ...                           dimension=3)

    Now associate id with Compounds for basis atoms and replicate 3x3x3
    >>> cscl_dict = {'Cl' : chlorine, 'Cs' : cesium}
    >>> cscl_compound = cscl_lattice.populate(x=3, y=3, z=3,
    ...                                       compound_dict=cscl_dict)

    A multi-Compound basis was created and replicated. For each unique basis
    atom position, a separate entry must be completed for the basis_atom
    input.

    Generating FCC Copper cell with lattice_vectors instead of angles
    >>> import mbuild as mb
    >>> copper = mb.Compound(name='Cu')
    >>> lattice_vector = ( [1, 0, 0], [0, 1, 0], [0, 0, 1])
    >>> spacings = [.36149, .36149, .36149]
    >>> copper_locations = [[0., 0., 0.], [.5, .5, 0.],
    ...                     [.5, 0., .5], [0., .5, .5]]
    >>> basis = {'Cu' : copper_locations}
    >>> copper_lattice = mb.Lattice(spacings, dimension=3,
    ...                           lattice_vectors=lattice_vector,
    ...                           basis_atoms=basis)
    >>> copper_dict = {'Cu' : copper}
    >>> copper_cell = copper_lattice.populate(x=3, y=3, z=20,
    ...                                       compound_dict=copper_dict)

    TODO(Justin Gilmer) : Print function to display info about Lattice (repr)
    TODO(Justin Gilmer) : inheritance(Cubic, orthorhombic, hexangonal)
    TODO(Justin Gilmer) : orientation functionality
    """

    def __init__(self, lattice_spacings, dimension=None,
                 lattice_vectors=None, basis_atoms=None,
                 angles=None):
        super(Lattice, self).__init__()
        self.lattice_spacings = None
        self.dimension = None
        self.lattice_vectors = None
        self.basis_atoms = dict()
        self.angles = None
        self.past_lat_vecs = deque()
        self.past_lat_vecs.append(lattice_vectors)
        self.redo_lat_vecs = deque()
        self._sanitize_inputs(lattice_vectors=lattice_vectors,
                              dimension=dimension,
                              lattice_spacings=lattice_spacings,
                              basis_atoms=basis_atoms,
                              angles=angles)

    def _sanitize_inputs(self, lattice_vectors, dimension,
                         lattice_spacings, basis_atoms, angles):
        """Check for proper inputs and set instance attributes.

        validate_inputs takes the data passed to the constructor by the user
        and will ensure that the data is correctly formatted and will then
        set its instance attributes.

        validate_inputs checks that dimensionality is maintained,
        the unit cell is right handed, the area or volume of the unit cell
        is positive and non-zero for 2D and 3D respectively, lattice spacings
        are provided, basis vectors do not overlap when the unit cell is
        expanded.

        Exceptions Raised
        -----------------
        TypeError : incorrect typing of the input parameters.

        ValueError : values are not within restrictions.
        """

        self._validate_dimension(dimension)
        self._validate_lattice_spacing(lattice_spacings, self.dimension)

        if angles and lattice_vectors:
            raise ValueError('Overdefined system: angles and lattice_vectors '
                             'provided. Only one of these should be passed.')
        if angles:
            self._validate_angles(angles, self.dimension)
            self.lattice_vectors = self._from_lattice_parameters(
                self.angles, self.dimension)
        else:
            self._validate_lattice_vectors(lattice_vectors, self.dimension)

        self._validate_basis_atoms(basis_atoms, self.dimension)

    def _validate_dimension(self, dimension):
        """Ensure that dimension input is correct.

        _validate_dimension will check for that the dimensionality
        passed to the constructor is a proper input.

        If the dimensionality is None, the default value is 3,
        or the user can specify 1D or 2D.

        If _validate_dimension cannot convert the passed in value to an int,
        or if the dimension is <1 or >3, a ValueError will be raised.

        Exceptions Raised
        -----------------
        ValueError : Incorrect typing of the input parameter.
        """
        if dimension is None:
            dimension = 3
        else:
            dimension = int(dimension)
        if dimension < 1 or dimension > 3:
            raise ValueError('Incorrect dimensions: {} is not a proper '
                             'dimension. 1, 2, or 3 are acceptable.'
                             .format(dimension))
        self.dimension = dimension

    def _validate_lattice_spacing(self, lattice_spacings, dimension):
        """Ensure that lattice spacing is provided and correct.

        _validate_lattice_spacing will ensure that the lattice spacings
        provided are acceptable values and dimensionally constant.

        Exceptions Raised
        -----------------
        ValueError : Incorrect lattice_vectors input
        """
        if lattice_spacings:
            lattice_spacings = np.asarray(lattice_spacings, dtype=float)
            if np.shape(lattice_spacings) != (dimension, ):
                raise ValueError('Lattice spacings should be a vector of '
                                 'size:({},). Please include lattice spacings '
                                 'for each available dimension.'
                                 .format(dimension))
        else:
            raise ValueError('Lattice Spacing Issue: None provided, '
                             'must provide lattice spacings matching '
                             'the dimension ({}) of the system.'
                             .format(dimension))
        if np.any(lattice_spacings <= 0.0):
            raise ValueError('Negative or zero lattice spacing value. One of '
                             'the spacings {} is negative or 0.'
                             .format(lattice_spacings))
        self.lattice_spacings = lattice_spacings

    def _validate_angles(self, angles, dimension):
        if angles:
            for index, value in enumerate(angles):
                angles[index] = float(value)
            if (len(angles), dimension) == (3, 3):
                if sum(angles) < 360.0 or sum(angles) > -360.0:
                    for theAngle in angles:
                        if(theAngle != 180.0 and theAngle != 0.0):
                            pass
                        else:
                            raise ValueError('Angles cannot be 180.0 or '
                                             '0.0.')
                else:
                    raise ValueError('Angles sum to a value greater than '
                                     '360.0 or less than -360.0.')

                for subset in it.permutations(angles, 3):
                    if not subset[0] < sum(angles) - subset[0]:
                        raise ValueError('Each angle provided must be less '
                                         'than the sum of the other two '
                                         'angles. {} is greater.'
                                         .format(subset[0]))
                self.angles = angles

            elif len(angles) == 1 and dimension == 2:
                for theAngle in angles:
                    if (theAngle != 180.0 and theAngle != 0.0 and
                            theAngle < 180.0 and theAngle > -180.0):
                        pass
                    else:
                        raise ValueError('Angle incorrectly defined. {} '
                                         'does not follow the proper '
                                         'guidelines for a bravais angle. '
                                         .format(theAngle))
                self.angles = angles
            else:
                raise ValueError('Incorrect amount of angles provided for '
                                 'dimension {}. Recieved {} angles.'
                                 .format(dimension, len(angles)))

    def _validate_lattice_vectors(self, lattice_vectors, dimension):
        """Ensure that the lattice_vectors are reasonable inputs.

        """
        if lattice_vectors is None:
                lattice_vectors = np.identity(dimension, dtype=float)
        else:
            lattice_vectors = np.asarray(lattice_vectors, dtype=float)
            shape = np.shape(lattice_vectors)

            if (dimension, dimension) != shape:
                raise ValueError('Dimensionality of lattice_vectors is '
                                 ' of shape {} not {}.'
                                 .format(shape, (dimension, dimension)))
            if dimension > 1:
                det = np.linalg.det(lattice_vectors)
                if abs(det) == 0.0:
                    raise ValueError('Co-linear vectors: {}'
                                     'have a determinant of 0.0. Does not '
                                     'define a unit cell.'
                                     .format(lattice_vectors))

                if det <= 0.0:
                    raise ValueError('Negative Determinant: the determinant '
                                     'of {} is negative, indicating a left-'
                                     'handed system.' .format(det))
        self.lattice_vectors = lattice_vectors

    def _validate_basis_atoms(self, basis_atoms, dimension):
        if basis_atoms is None:
            basis_atoms = {}
            basis_atoms = {'default': [[0. for x in range(dimension)]]}
        elif isinstance(basis_atoms, dict):
            pass
        else:
            raise TypeError('Incorrect type, basis_atoms is of type {}, '
                            'Expected dict.'.format(type(basis_atoms)))

        for name in basis_atoms.keys():
            positions = basis_atoms[name]
            for pos in positions:
                location_check = []
                if len(pos) != dimension:
                    raise ValueError("Incorrect basis atom position size. "
                                     "Basis atom {} was passed with location "
                                     "{}, which is inconsistent with the "
                                     "dimension {}.".format(name, pos,
                                                            dimension))
                if pos is None:
                    raise ValueError("NoneType passed, expected float. "
                                     "None was passed in as position for {}."
                                     .format(name))

                location_check = [coord for coord in pos if coord is None or coord >= 1. or coord < 0.]
                if len(location_check) != 0:
                    raise ValueError("Incorrect coordinate value for basis. "
                                     "Basis {}, was passed coordinates {}. "
                                     "The coordinates {}, were either < 0, or"
                                     " > 1.".format(name, pos, location_check))

        self.basis_atoms = self._check_for_overlap(basis_atoms, dimension)

    def _check_for_overlap(self, basis_atoms, dimension):

        overlap_dict = defaultdict(list)
        num_iter = 3
        for name in basis_atoms.keys():
            positions = basis_atoms[name]
            for pos in positions:
                for offsets in it.product(range(num_iter), repeat=dimension):
                    offset_vector = tuple((v + offset for v, offset in zip(pos, offsets)))
                    overlap_dict[offset_vector].append((pos))

        for key, val in overlap_dict.items():
            if len(val) > 1:
                raise ValueError('Overlapping Basis Vectors: Basis '
                                 'vectors overlap when the unit cell is '
                                 'expanded to {}. This is an incorrect '
                                 'perfect lattice. The offending '
                                 'vectors are: {}'
                                 .format(key, val))
        return basis_atoms

    def _from_lattice_parameters(self, angles, dimension):
        """Convert Bravais lattice parameters to lattice vectors.

        _from_lattice_parameters will generate the lattice vectors based on
        the parameters necessary to build a Bravais Lattice.

        This was adapted from the ASE triclinic.py lattice parameter code.

        S. R. Bahn and K. W. Jacobsen
        An object-oriented scripting interface to a
        legacy electronic structure code Comput. Sci. Eng., Vol. 4, 56-66, 2002

        Parameters
        ----------
        angles : list-like, required
            Angles of bravais lattice.
        dimension : integer, required
            Dimensionality of system, can only be 2 or 3.
        """
        if dimension is 3:
            (alpha, beta, gamma) = angles

            degree = np.pi / 180.0
            cosa = np.cos(alpha * degree)
            cosb = np.cos(beta * degree)
            sinb = np.sin(beta * degree)
            cosg = np.cos(gamma * degree)
            sing = np.sin(gamma * degree)
            lattice_vec = ([1, 0, 0],
                           [cosg, sing, 0],
                           [cosb, (cosa - cosb * cosg) / sing,
                            np.sqrt(sinb**2 - ((cosa - cosb * cosg) / sing)**2)])
        else:
            alpha = angles
            degree = np.pi / 180.0
            cosa = np.cos(alpha * degree)
            sina = np.sin(alpha * degree)
            lattice_vec = ([1, 0], [cosa, sina])

        return lattice_vec

    def populate(self, compound_dict=None, x=1, y=1, z=1):
        """Expand lattice and create compound from lattice.

        populate will expand lattice based on user input. The user must also
        pass in a dictionary that contains the keys that exist in the
        basis_dict. The corresponding Compound will be the full lattice
        returned to the user.

        If no dictionary is passed to the user, Dummy Compounds will be used.

        Parameters
        ----------
        x : int, optional, default=1
            How many iterations in the x direction.
        y : int, optional, default=1
            How many iterations in the y direction.
        z : int, optional, default=1
            How many iterations in the z direction.
        compound_dict : dictionary, optional, default=None
            Link between basis_dict and Compounds.

        Exceptions Raised
        -----------------
        ValueError : incorrect x,y, or z values.
        TypeError : incorrect type for basis vector

        Call Restrictions
        -----------------
        Called after constructor by user.
        """
        error_dict = {0:'X', 1:'Y', 2:'Z'}

        # padded for Compound compatibility
        cell_edges = [edge[0] for edge in it.zip_longest(self.lattice_spacings, range(3), fillvalue=0.0)]

        for replication_amount in x, y, z:
            if replication_amount is None:
                raise ValueError('Attempt to replicate None times. '
                                 'None is not an acceptable replication amount, '
                                 '1 is the default.')

        for replication_amount, index in zip([x, y, z], range(3)):
            if replication_amount < 1:
                raise ValueError('Incorrect populate value: {} : {} is < 1. '
                                 .format(error_dict[index], replication_amount))

        if self.dimension == 2:
            if z > 1:
                raise ValueError('Attempting to replicate in Z. '
                                 'A non-default value for Z is being '
                                 'passed. 1 is the default value, not {}.'
                                 .format(z))
        elif self.dimension == 1:
            if (y > 1) or (z > 1):
                raise ValueError('Attempting to replicate in Y or Z. '
                                 'A non-default value for Y or Z is being '
                                 'passed. 1 is the default value.')
        else:
            pass

        if ((isinstance(compound_dict, dict)) or (compound_dict is None)):
            pass
        else:
            raise TypeError('Compound dictionary is not of type dict. '
                            '{} was passed.'.format(type(compound_dict)))

        cell = defaultdict(list)
        [a, b, c] = cell_edges
        for key, locations in self.basis_atoms.items():
            for coords in range(len(locations)):
                for replication in it.product(range(x), range(y), range(z)):
                    tmpx = (locations[coords][0] + replication[0]) * a

                    try:
                        tmpy = (locations[coords][1] + replication[1]) * b
                    except IndexError:
                        tmpy = 0.0

                    try:
                        tmpz = (locations[coords][2] + replication[2]) * c
                    except IndexError:
                        tmpz = 0.0

                    tmp_tuple = tuple((tmpx, tmpy, tmpz))
                    cell[key].append(((tmp_tuple)))

        ret_lattice = mb.Compound()
        if compound_dict is None:
            for key_id, all_pos in cell.items():
                particle = mb.Particle(name=key_id, pos=[0, 0, 0])
                for pos in all_pos:
                    particle_to_add = mb.clone(particle)
                    mb.translate(particle_to_add, list(pos))
                    ret_lattice.add(particle_to_add)
        else:
            for key_id, all_pos in cell.items():
                if isinstance(compound_dict[key_id], mb.Compound):
                    compound_to_move = compound_dict[key_id]
                    for pos in all_pos:
                        tmp_comp = mb.clone(compound_to_move)
                        mb.translate(tmp_comp, list(pos))
                        ret_lattice.add(tmp_comp)
                else:
                    err_type = type(compound_dict.get(key_id))
                    raise TypeError('Invalid type in provided Compound dictionary. '
                              'For key {}, type: {} was provided, '
                              'not mbuild.Compound.'.format(key_id, err_type))
        return ret_lattice
    
    
    def rotate_lattice(self, lat, new_view, miller_directions = False, new_face = None,
                       by_angles = False, degrees = False,
                       rot_by_lat_vecs= False):
        """Use this to rotate the lattice once populated.*****
        
        *****add back in future point option. this can be achived if provided 
        with a point that will be the new origin, a point that will lie on the
        new xaxis and a point that will lie in the xy plane. this can also be done
        by defining the origin+a point on the y and the xy or the same for the z
        axis. see x/y/z_axis_transform under coordinate_transform. There is also
        the AxisTransform class but it does not seem quite as effective. 
        
        
        *******make sure to include the option that allows the user to supply 3
        angles, how much to rotate about each axis respectively. this option is
        only valid when axis = False. this result will be acheived through the 
        RotationAroundXfunction in coordinate transform.
        
        do an example with each method
        
        Parameters
        ----------
        new_view : list, required, defaults to axis+angle+point(AAP) option
            Defines the new orientation of the lattice. Accepts a list of 3 arguments. 
            It will only accept 2 arguments if the axis+angle+point option is chosen,
            where the third argument will default to [0,0,0]. The AAP option
            is chosen when and only when both miller_directions and future_points are False.
            In the AAP option, the first argument of the list is a 3D vector in the
            form [x,y,z] which serves as an axis the lattice will rotate about. The 
            second argument is an angle, in degrees, (int or float) that the lattice 
            will rotate clockwise (the line of sight points the same direction as the
            axis vector) about the specified axis. The third argument is a point in 
            the form [x,y,z] that the axis of rotation must pass through.
                   
         miller_directions : boolean, optional, default = False
            When assigned True, no longer in the AAP option. Now the new orientation is
            defined by Miller coordinates. Must still feed new_view a list of 3 arguments,
            now containing Miller coorindates in the form...............
                 
                
         degrees : boolean, option, default = False
            This parameter can only equal True when the by_angles option is also True, or when 
            the AA. 
            This parameter when True changes the angle(s) supplied in new_view from radians to
            degrees.
        
        new_face : accepts str, optional, default = None
            Only accepts None, or a specified axis, 'x', 'y', or 'z', all case insenstive. When
            new_face is not None, the new face is defined by three points, passed as 3 lists
            np.ndarrays (of size 3) or mb.Compounds that lie within the list passed for new_view. 
            In the case where the user inputs 'x', they must also pass ..........
                    if the user passes three points that already lie on the specified plane new_face
                    will mirror the crystal
        
        by_angles : boolean, optional, default = False
            If set to True, the user must provide a list of size 3 to new_view containing
            the values of how much the user wants to rotate the crystal by, about each axis,
            in the order x,y,z
                   
                   
                   
                make sure to describe output   
                
                   do examples with miller_directions, and aap
                   
                Errors to raise :
                       TypeErrors
            -most Errors for new_view are checked in the conditionals except type
            -TypeErrors for the other input values are checked at the beginning
            -Right handedness and these things are checked inside of miller_directions
            -The process/conditional for new_face is checked and carried out in the 
            TypeError section at the beginning
            
    
        """
        if not isinstance(lat, mb.Compound):
            raise TypeError('lat must be of custom type mb.Compound. '
                            'Type: {} was passed.'.format(type(lat)))
        if not isinstance(new_view, list):
            raise TypeError('new_view must be of type list. '
                            'Type: {} was passed.'.format(type(new_view)))
        if not isinstance(miller_directions, bool):
            raise TypeError('miller_directions must be of type bool. '
                            'Type: {} was passed.'.format(type(miller_directions)))
        if not isinstance(by_angles, bool):
            raise TypeError('by_angles must be of type bool. '
                            'Type: {} was passed.'.format(type(by_angles)))
        if not isinstance(degrees, bool):
            raise TypeError('degrees must be of type bool. '
                            'Type: {} was passed.'.format(type(degrees)))
            #the next part must be done before new_face checks.
        #grab the old lattice vectors
        #create a list to store the most current location of each lattice vector lattice vectors in
        updated_lat_vecs = [kk for kk in self.lattice_vectors]
        
        if new_face:
            if self.dimension != 3: # sketchy on this one
                raise ValueError("The new_face option only works with 3D objects")
            elif not isinstance(new_face, str):
                raise TypeError('new_face must be of type None or str. '
                            'Type: {} was passed.'.format(type(new_face))) 
            elif miller_directions or by_angles:
                raise ValueError('Overdefined system: only zero or one of the following'
                                ' is allowed to be a non-falsy value: miller_directions, by_angles, '
                                'new_face.')            
                # this next conditional commands carry out the type error and execution of
                # new_face option
            new_face = new_face.lower()
            face_dict = {'x' : x_axis_transform, 'y' : y_axis_transform, 
                         'z' : z_axis_transform}
            if new_face not in face_dict:
                raise ValueError("new_face only accepts None, 'x', 'y', or 'z'."
                                'The strings are case insensitive.')
            #now we check the validity of the new_view passed
            if not isinstance(new_view, list):
                raise TypeError('When new_face option is selected, new_view must '
                                'be a list of 3 mb.Compounds, or of 3 np.ndarrays, ' 
                                'lists, or tuples (each size 3)')
            if len(new_view) != 3:
                raise ValueError('When new_face option is selected, new_view must be'
                                'a list of size 3.')
            for indy in new_view:
                if not isinstance(indy, (np.ndarray, mb.Compound, list)):
                    raise TypeError('When new_face option is selected, new_view must'
                                   'be a list of 3 np.ndarrays (each size 3) or of 3'
                                   ' mb.Compounds, or of 3 lists')
            if degrees:
                warn('degrees passed as True although no data were passed with it that require'
                     ' the degrees specification. Unused parameter, calulations unaffected.')   
            for part in lat.children:
                #now we write the code to rotate that hoe.
                #this may be very wrong 
                face_dict[new_face](part, new_view[0], new_view[1], new_view[2])
            #update the lattice vectors
            for jj in range(len(updated_lat_vecs)):
#                 print('___________________________________________________')
#                 print("____________________________________________________")
#                 print(updated_lat_vecs)
#                 print('_')    
                dummy = mb.Compound()
                dummy.pos = updated_lat_vecs[jj]
                face_dict[new_face](dummy, new_view[0], new_view[1], new_view[2])
                updated_lat_vecs[jj] = dummy.pos
                #print(updated_lat_vecs)
                #test this in the window below with print statements 

                
                #########
            return 
        self.redo_lat_vecs.clear()
                    
                   
        standard_option = False
        if not miller_directions and not by_angles:
            standard_option = True
            
        if degrees and not (by_angles or standard_option):
            warn('degrees passed as True although not data were passed with it that require'
                ' the degrees specification. Unused parameter, calculations unaffected.')
               
        #in each conditional ensure not overdefinted and
        #also that types/values are valid for new_view. 
        
        
        if by_angles:
            #this wont work with 2D I dont think
            if miller_directions:
                raise ValueError('Overdefined system: only zero or one of the following'
                                ' is allowed to be a non-falsy value: miller_directions, by_angles, '
                                'new_face.') 
            elif True: # type check new_view
                pass
            if degrees:
                new_view = [np.pi*jj/180 for jj in new_view]
            by_angles_list = [RotationAroundX, RotationAroundY, RotationAroundZ]
            for ii in range(len(new_view)):
                #need something here about how to track new lat vecs
                updated_lat_vecs = [by_angles_list[ii](new_view[ii]).apply_to(jj)[0] for jj in updated_lat_vecs]
                for parti in lat.children:
                    parti.pos = by_angles_list[ii](new_view[ii]).apply_to(parti.pos)[0]
            
                        
        
            
        elif miller_directions:
            
            #consider modifying so that the 
            
            # rename to miller_orientations or miller_directions
            
            #include an error message that new_view only accepts a list of lists
            #if the user is only interested in passing the miller_directions indicies in 2 or fewer
            #directions, for example just the Z direction, the user may pass a list of 
            #lists in the form [[],[],[#,#,#]]. this will cause the lattice to rotate
            # in a way so that only the Z axis is aligned in this configuration.
            #also include a bit in the function description about this feature
                          
            #make sure to check for handedness and things of this sort
            
            # also check the 2D case................not 2d compatible?
            empty_tracker = 0
            for jj in range(len(new_view)):
                if not isinstance(new_view[jj], list):
                    if isinstance(new_view[jj], np.ndarray):
                        new_view[jj] = new_view[jj].tolist()
                    elif isinstance(new_view[jj], tuple):
                        new_view[jj] = list(new_view[jj])
                    else:
                        raise TypeError('When miller_directions option is selected, new_view must be either a '
                                        'list of length 3, made up of lists, tuples, or np.ndarrays, '
                                        'each containing either 3D Miller directions OR an empty list, '
                                        'although, new_view may only have one empty argument. Type: {}' 
                                        'was passed.'.format(type(new_view[jj])))
                if not new_view[jj]:
                    empty_tracker+=1
                    if empty_tracker > 1:
                        raise ValueError('When miller_directions option is selected, user is only '
                                         "able to leave maximum 1 of new_view's arguments "
                                         'empty')
                    to_be_crossed_dict = {0 : [1,2], 1 : [0,2], 2 : [0,1]}
                    to_be_crossed = to_be_crossed_dict[jj]
                    missing_vector_index = jj
                else:
                    for ii in new_view[jj]:
                        if not isinstance(ii, (float, int)):
                            raise TypeError('When miller_directions option is selected, the lists or '
                                            'numpy ndarrays inside the new_view list must '
                                            'contain either all floats or ints describing '
                                            '3D Miller directions. {} was passed.'
                                            .format(type(ii)))
                    new_view[jj] /= np.linalg.norm(new_view[jj])

            if empty_tracker == 1:
                new_view[missing_vector_index] = np.cross(new_view[to_be_crossed[0]],
                                         new_view[to_be_crossed[1]])
                # should I error check this above piece? i'm thinking no
                new_view[missing_vector_index] /= np.linalg.norm(new_view[missing_vector_index])
                handed = np.linalg.det(new_view)
                if handed == 0:
                    raise ValueError('Co-linear vectors. The miller_directions directions entered are '
                                     'not valid, as they have a determinant of 0.')
                elif handed < 0:
                    new_view[missing_vector_index] *= -1
                    handed = np.linalg.det(new_view)
                    if handed < 0:
                        raise ValueError('The miller_directions entered are not valid. Check '
                                         'orthagonality.')
                if 1e-14 < abs(handed - 1):
                    warn('The determinant of the rotation matrix (miller directions) '
                         'varies by more than 1e-14 from 1, this may be indicative of '
                         'impractical miller directions.')
                    print(new_view)
                    print(handed)
            else:
                handed = np.linalg.det(new_view)
                if handed == 0:
                    raise ValueError('Co-linear vectors. The miller_directions directions entered are '
                                     'not valid, as they have a determinant of 0.')
                elif handed < 0:
                    raise ValueError('The miller_directions directions entered are not valid, as they'
                                     'have a negative determinant, thus a left-handed system.')
                elif 1e-14 < abs(handed - 1):
                    warn('The determinant of the rotation matrix (miller directions) '
                         'varies by more than 1e-14 from 1, this may be indicative of '
                         'impractical miller directions.')
                    print(new_view)
                    print(handed)
            rotation_matrix = new_view
            for part in lat.children:
                part.pos = np.matmul(rotation_matrix, part.pos)
            updated_lat_vecs = [np.matmul(rotation_matrix, nn) for nn in updated_lat_vecs]


            ####### this is a previous version of miller (below) that may be used later on.
            # now we are all checked up and can proceed
#             for ii in range (len(new_view)):
#                 #this next conditional statement checks for the case which the user 
#                 #chooses to specify only one miller_directions index
#                 if new_view[ii] == []:
#                     continue
#                 orthag = np.cross(np.array(new_view[ii]), updated_lat_vecs[ii]) # which goes first??????
#                 #back calculating theta (radians) using angle(), defined in coordinate_transform
#                 theta = angle(np.array(new_view[ii]), updated_lat_vecs[ii])
#                 #check determinates of new lattice vectors for handedness, this part is just for me
#                 # to see if the whole thing actually works
#                 updated_lat_vecs = [Rotation(theta, orthag).apply_to(jj)[0] for jj in updated_lat_vecs]
#                 for part in lat.children:
#                     part.pos = Rotation(theta, orthag).apply_to(part.pos)[0] 

        
        
        elif standard_option:
            #check new_view  
            if len(new_view) != 2:
                raise ValueError('When using the default standard_option for rotate_lattice,'
                                 ' new_view must be a list of size 2. The size varies'
                                 ' depending on the option selected.')
            if not isinstance(new_view[0], (list, np.ndarray, tuple)):
                raise TypeError('When using the default standard_option for rotate_lattice,'
                                'the first index of new_view must be either a list, tuple or '
                                'a numpy ndarray of {}D coordinates. User passed {}.'
                                .format(self.dimension, type(new_view[0])))
            if len(new_view[0]) != self.dimension:
                raise ValueError('The first index of new_view must be of size {}'
                                 ' when using the default, standard_option'.format(self.dimension))
            if not isinstance(new_view[1], (float, int)):
                raise TypeError('When using the default, standard_option, the second index'
                                ' of new_view must either be of type int or float.'
                                'Type {} was passed.'.format(type(new_view[1])))
            axis = np.array(new_view[0])
            if degrees:
                theta = np.pi*new_view[1]/180
            else:
                theta = new_view[1]
            updated_lat_vecs = [Rotation(theta, axis).apply_to(jj)[0] for jj in updated_lat_vecs]
            for part in lat.children: 
                part.pos = Rotation(theta, axis).apply_to(part.pos)[0]
        else:
            #this seems superfluous but i feel like i'm missing something...placeholder?
            raise ValueError('underdefined system')
            
        self.lattice_vectors = np.array(updated_lat_vecs) 
        self.past_lat_vecs.append(self.lattice_vectors)
        # new face still does not update the past lat vecs 
             
    def mirror(self, cmpnd, about):
        """ 
        
        Parameters:
        ------------
        about : str, case insensitive, order insensitive.
            If dimensions are 2D, about is of length 2, """
        # consider adding a way to track if the lattice has been rotated
        
        # look into the use rot_by_lat_vecs option above and how it relates here/ if necessary
        
        # dont forget to track changing axes
        
        #still iffy on the necessity of the keep lattice vecs argument 
        
        if not isinstance(cmpnd, mb.Compound):
            raise TypeError('This lattice method must be applied to a compound.'
                           'User passed {} instead.'.format(type(cmpnd)))        
        if not isinstance(about, str):
            raise TypeError('about only accepts strings. User passed: {}.'.format(type(about)))
        if len(about) != (self.dimension - 1):
            raise ValueError('about must be a string of length {} when dimensions are {}'
                            'User passed string of length {}.'
                             .format((self.dimension - 1), self.dimension, len(about)))
        about = about.lower()
        if self.dimension == 2 and about == 'z':
            raise ValueError('This lattice is 2D this it cannot be reflected about the z-axis')
        str_dict = {'x' : 0, 'y' : 1, 'z' : 2}
        w = np.ones(self.dimension).tolist()
        for letta in about:
            if letta not in str_dict.keys():
                raise ValueError('String not recognized. For {}D lattices, only{} (not case '
                                 'or order sensitive) are valid arguments for about '
                                 'parameter.'.format(self.dimension,
                                                     ' x, y, z'[:(3*self.dimension)]))
            else:
                w[str_dict[letta]] = 0
        which_flip = w.index(1)
        updated_lat_vecs = [kk for kk in self.lattice_vectors]
        
        
#         if self.rotated:
#             if which_flip == 1:
#                 #xz
#                 self.rotate_lattice(lat = cmpnd,
#                               new_view = [[0,0,0],[0,0,1],[0.5,0,0.5]], miller = False, 
#                               by_angles = False, new_face = 'z', degrees = True,
#                               keep_lat_vecs = False) # include all flags 
#             else:
#                 #xy
#                 self.rotate_lattice(lat = cmpnd,
#                               new_view = [[0,0,0],[0,1,0],[0.5,0.5,0]], miller = False, 
#                               by_angles = False, new_face = 'x', degrees = True,
#                               keep_lat_vecs = False) # include all flags 
#                 if which_flip == 0:
#                     #YZ
#                     self.rotate_lattice(lat = cmpnd, new_view = [0,180,0], miller = False,
#                                         by_angles = True, new_face = None, degrees = True,
#                                        keep_lat_vecs = False) # include all flags
#         else:
#             for part in cmpnd.children:
#                 part.pos[which_flip] = -1*part.pos[which_flip]
        for part in cmpnd.children:
            part.pos[which_flip] *= -1
        for ii in range(self.dimension):
            updated_lat_vecs[ii][which_flip] *= -1
        # make sure to update past lattice vecs 
        
            
            
    def undo_rotation(self, cmpnd, OG = False):
        """rotate back to original orientation or just a one."""
        if len(self.past_lat_vecs) == 1:
            raise ValueError('Cannot undo since this is the original lattice orientation.')
        start = self.past_lat_vecs.pop()
        self.redo_lat_vecs.append(start)
        if OG:
            while len(self.past_lat_vecs) > 1:
                self.redo_lat_vecs.append(self.past_lat_vecs.pop())
        destination = self.past_lat_vecs[-1]
        self.lattice_vectors = destination
        # and now we rotate
        
        R = np.matmul(destination, np.linalg.inv(start))
        for ii in R:
            ii /= np.linalg.norm(ii)
        for part in cmpnd.children:
            part.pos = np.matmul(R, part.pos)
        
                
    def redo_rotation(self, cmpnd, redo_all = False):
        """"""
        if len(self.redo_lat_vecs) == 0:
            raise ValueError('Cannot redo, this is most current rotation.')
        start = self.past_lat_vecs[-1]
        
        if redo_all:
            while len(self.redo_lat_vecs) > 0:
                self.past_lat_vecs.append(self.redo_lat_vecs.pop())
        else:
            self.past_lat_vecs.append(self.redo_lat_vecs.pop())
        destination = self.past_lat_vecs[-1]
        self.lattice_vectors = destination
        # now we rotate
        
        R = np.matmul(destination, np.linalg.inv(start))
        for ii in R:
            ii /= np.linalg.norm(ii)
        for part in cmpnd.children:
            part.pos = np.matmul(R, part.pos)
                
                

                
print('issa vibe')


# In[ ]:

import mbuild as mb

dim = 3
cscl_lengths = [.4123, .4123, .4123]
cscl_vectors = [[1,0,0], [0,1,0], [0,0,1]]
cscl_basis = {'Cs':[[0, 0, 0]], 'Cl':[[.5, .5, .5]]}
cscl_lattice = Lattice(cscl_lengths, dimension=dim,
                                lattice_vectors=cscl_vectors, basis_atoms=cscl_basis)
cs = mb.Compound(name='Cs')
cl = mb.Compound(name='Cl')
cscl_dict = {'Cs':cs, 'Cl':cl}
cscl_crystal = cscl_lattice.populate(compound_dict=cscl_dict, x=3, y=3, z=3)
#cscl_crystal = cscl_lattice.populate(compound_dict=cscl_dict, x=2, y=2, z=2)



import numpy as np
dum = []
for part in cscl_crystal:
    if np.array_equal([0,0,0], part.pos):
#         print(part.pos)
#         print(type(cscl_crystal))
#         print(part.name)
        part.name='Rb'
        
    if np.sum([1.0308,1.0308,1])<= np.sum(part.pos):
#         print(part.pos)
#         print(type(cscl_crystal))
#         print(part.name)
        part.name='O'
        
    if part.pos[0] == 0 and part.pos[1] == 0 and .1<= part.pos[2]<=.5:
#         print(part.pos)
#         print(type(cscl_crystal))
#         print(part.name)
        part.name='N'
    dum.append([part.name, part.pos])
#print(dum)
dum = []
#cscl_crystal.save('cscl_crystal_OG_labeled_3x3x3.mol2', overwrite = True)
#print("da OG")
print('OG')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
print(' ')
OG_crystal = mb.compound.clone(cscl_crystal)


cscl_lattice.rotate_lattice(lat= cscl_crystal, new_view= [[1,1,1], 120], degrees= True)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
# print(dum)
# dum = []
#cscl_crystal.save('cscl_crystal_AA_120_3x3x3.mol2', overwrite = True)
#print("1st rotation")
print('after first rot')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
print(" ")
rot1_crystal = mb.compound.clone(cscl_crystal)

cscl_lattice.rotate_lattice(lat= cscl_crystal, new_view= [[1,1,1], 120], degrees= True)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
#print(dum)
#dum = []
# not a correct name 
#cscl_crystal.save('cscl_crystal_after_miller_neg211_111_01neg1_from_AA_120_3x3x3.mol2', overwrite = True)
#print("2nd rotation")
print('after second rotate')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
print(' ')
rot2_crystal = mb.compound.clone(cscl_crystal)


cscl_lattice.undo_rotation(cmpnd= cscl_crystal)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
#print(dum)
# dum = []
#cscl_crystal.save('cscl_crystal_undo1.mol2', overwrite = True)
#print("undo 1")
print('after first undo')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
undo1_crystal = mb.compound.clone(cscl_crystal)

cscl_lattice.undo_rotation(cmpnd=cscl_crystal)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
# print(dum)
# dum = []
# cscl_crystal.save('cscl_crystal_undo2.mol2', overwrite = True)
# print("undo 2")
print('after 2nd undo')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
undo2_crystal = mb.compound.clone(cscl_crystal)

cscl_lattice.redo_rotation(cmpnd = cscl_crystal)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
# print(dum)
# dum = []
# cscl_crystal.save('cscl_crystal_redo1.mol2', overwrite = True)
# print("redo 1")
print('after first redo')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
redo1_crystal = mb.compound.clone(cscl_crystal)


cscl_lattice.redo_rotation(cmpnd = cscl_crystal)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
# print(dum)
# dum = []
# cscl_crystal.save('cscl_crystal_redo2.mol2', overwrite = True)
# print("redo 2")
print('after second redo')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
redo2_crystal = mb.compound.clone(cscl_crystal)

cscl_lattice.undo_rotation(cmpnd = cscl_crystal, OG= True)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
# print(dum)
# dum = []
# cscl_crystal.save('cscl_crystal_undoall.mol2', overwrite = True)
# print("undo all")
print('after undo all')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
undoall_crystal = mb.compound.clone(cscl_crystal)

cscl_lattice.redo_rotation(cmpnd = cscl_crystal, redo_all = True)
# for part in cscl_crystal:
#     dum.append([part.name, part.pos])
# print(dum)
# dum = []
# cscl_crystal.save('cscl_crystal_redoall.mol2', overwrite = True)
# print("redo all")
print('after redo all')
print(cscl_lattice.lattice_vectors)
print('lattice vecs')
print(cscl_lattice.past_lat_vecs)
print("past lat vecs")
print(cscl_lattice.redo_lat_vecs)
print('redo lat vecs')
redoall_crystal = mb.compound.clone(cscl_crystal)


print("Matches:")
print("OG, undo all, undo 2")
print("redo all, rotation 2, redo 2 ")
print("rotation 1, redo 1, undo 1")


# In[ ]:

for part1, part2 in zip(rot2_crystal, redoall_crystal):
    print(part1.pos)
    print(part2.pos)
    assert(np.allclose(part1.pos, part2.pos, atol=1e-15))
    print(' ')


# In[ ]:

import mbuild as mb
import nglview

#set up the dimensions and make the crystal
dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}

simple_cubic = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}

crystal_polonium = simple_cubic.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
toddy = simple_cubic
simple_cubic.rotate_lattice(lat = crystal_polonium, 
                              new_view = [[1,1,1],120], 
                              by_angles = False, new_face = None, degrees = True)
print(simple_cubic.lattice_vectors)
print(toddy.lattice_vectors)


# In[ ]:

#polonium crystal




import mbuild as mb
import nglview

#set up the dimensions and make the crystal
dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}

simple_cubic = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}

crystal_polonium = simple_cubic.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)

#now color the crystal
import numpy as np
for part in crystal_polonium:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'

    
    print(part)

# # now save it as is (the OG)
# crystal_polonium.save('polonium_crystal_OG.mol2', overwrite = True)
#crystal_polonium.save('polonium_crystal_OG_hoomd.mol2', overwrite = True)


##########################################################################
# # rotate it back
# #simple_cubic.rotate_lattice(lat = crystal_polonium, new_view = , miller = , by_angles = ,
#  #                          new_face = , degrees = True, keep_lat_vecs = False)
# crystal_polonium.save('polonium_crystal_OG_from_rot1_miller.mol2', overwrite = True)

"""other tests to do:

crystal_polonium.save('polonium_crystal_rot1_AA.mol2', overwrite = True) 
crystal_polonium.save('polonium_crystal_OG_from_rot1_AA___with_AA.mol2', overwrite = True)

crystal_polonium.save('polonium_crystal_rot1_new_face.mol2', overwrite = True)
crystal_polonium.save('polonium_crystal_OG_from_rot1_new_face__with_new_face.mol2', overwrite = True)

crystal_polonium.save('polonium_crystal_rot1_by_angles.mol2', overwrite = True)
crystal_polonium.save('polonium_crystal_OG_from_rot1_by_angles__with_by_angles.mol2', overwrite = True)
........the list goes on&on&on&on&on&on&on&on&on&on"""


#crystal_polonium.visualize()


# In[ ]:

#  rotate using axis angle (and back) polonium 
# must make cystal first
dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}
simple_cubicAA = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}
crystal_poloniumAA = simple_cubicAA.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
for part in crystal_poloniumAA:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'
    print(part)
#############################################################

simple_cubicAA.rotate_lattice(lat = crystal_poloniumAA, 
                              new_view = [[1,1,1],120], miller_directions = False, 
                              by_angles = False, new_face = None, degrees = True)
print('120')
for part in crystal_poloniumAA:
    print(part)
    
#crystal_poloniumAA.save('polonium_crystalAA_120.mol2', overwrite = True)


# rotate again 
simple_cubicAA.rotate_lattice(lat = crystal_poloniumAA,
                              new_view = [[1,1,1],120], miller_directions = False,
                              by_angles = False, new_face = None,
                              degrees = True)
print('240')
for part in crystal_poloniumAA:
    print(part)
#crystal_poloniumAA.save('polonium_crystalAA_2x120.mol2', overwrite = True)


#and again, should now be back to normal 
simple_cubicAA.rotate_lattice(lat = crystal_poloniumAA, 
                              new_view = [[1,1,1],120], miller_directions = False, 
                              by_angles = False, new_face = None,
                              degrees = True)
print('360')
for part in crystal_poloniumAA:
    print(part)
#crystal_poloniumAA.save('polonium_crystalAA_3x120.mol2', overwrite = True)


# In[ ]:

#rotate using angles, compare with AA

dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}
simple_cubicBA = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}
crystal_poloniumBA = simple_cubicBA.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
for part in crystal_poloniumBA:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'
        
##########################################

simple_cubicBA.rotate_lattice(lat = crystal_poloniumBA, 
                              new_view = [90,0,90], 
                              by_angles = True, new_face = None, degrees = True)
#crystal_poloniumBA.save('polonium_crystalBA.mol2', overwrite = True)


#####################################
simple_cubicAA = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}
crystal_poloniumAA = simple_cubicAA.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
for part in crystal_poloniumAA:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'
####################################
simple_cubicAA.rotate_lattice(lat = crystal_poloniumAA, 
                              new_view = [[1,1,1],120], 
                              by_angles = False, new_face = None, degrees = True)
######################

for part1,part2 in zip(crystal_poloniumAA,crystal_poloniumBA):
    print('120 ')
    print(part1)
    print('BA ')
    print(part2)
    print(" ")


# In[ ]:

# rotating using the new face method 

dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}
simple_cubicNF = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}
crystal_poloniumNF = simple_cubicNF.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
for part in crystal_poloniumNF:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'

##########################################

simple_cubicNF.rotate_lattice(lat = crystal_poloniumNF,
                              new_view = [[0,0,0],[0,1,0],[0.5,0.5,0]], 
                              by_angles = False, new_face = 'x', degrees = True)

for part in crystal_poloniumNF:
    print(part)
     
    

#crystal_poloniumNF.save('polonium_crystalNF.mol2', overwrite = True)


# In[ ]:

# use NF to reflect across XY

dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}
simple_cubicNF = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}
crystal_poloniumNF = simple_cubicNF.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
for part in crystal_poloniumNF:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'
    print(part)

##########################################

simple_cubicNF.rotate_lattice(lat = crystal_poloniumNF,
                              new_view = [[0,0,0],[0,1,0],[0.5,0.5,0]], 
                              by_angles = False, new_face = 'x', degrees = True)

#crystal_poloniumNF.save('polonium_crystalNF_reflect_XY.mol2', overwrite = True)



###################################################
# this second rotation puts it across the YZ
simple_cubicNF.rotate_lattice(lat = crystal_poloniumNF, new_view =  [0,180,0], by_angles = True,
                             degrees = True)

#crystal_poloniumNF.save('polonium_crystalNF_reflect_YZ.mol2', overwrite = True)


# In[ ]:

#uses NF to reflect across xz

dim = 3
edge_lengths = [.3359, .3359, .3359]
lattice_vecs = [[1,0,0], [0,1,0], [0,0,1]]
basis = {'origin':[[0,0,0]]}
simple_cubicNF = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}
crystal_poloniumNF = simple_cubicNF.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)
for part in crystal_poloniumNF:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'
        
##########################################

simple_cubicNF.rotate_lattice(lat = crystal_poloniumNF,
                              new_view = [[0,0,0],[0,0,1],[0.5,0,0.5]], 
                              by_angles = False, new_face = 'z', degrees = True)

for part in crystal_poloniumNF:
    print(part)


# In[ ]:

#     rotating using miller 




# simple_cubic.rotate_lattice(lat = crystal_polonium, new_view = [[-2,1,1], [1,1,1], [0,1,-1]], miller = True, by_angles = False,
#                             new_face = False, degrees = False, keep_lat_vecs = False)

simple_cubic.rotate_lattice(lat = crystal_polonium, new_view = [[0,1,0], [0,0,1], [1,0,0]], miller_directions = True, by_angles = False,
                            new_face = False, degrees = False)
#crystal_polonium.save('polonium_crystal_rot1_miller.mol2', overwrite = True)
for part in crystal_polonium:
    print(part.pos)


# In[ ]:

# rotatiom using the rotation matrix



#make crystal
simple_cubicRM = Lattice(edge_lengths, 
                          lattice_vectors=lattice_vecs, dimension=dim, 
                          basis_atoms=basis)
po = mb.Compound(name='Po')
compound_dictionary = {'origin':po}

crystal_poloniumRM = simple_cubicRM.populate(compound_dict=compound_dictionary, x=2, y=2, z=2)

#now color the crystal
import numpy as np
for part in crystal_poloniumRM:
    if np.array_equal([0,0,0], part.pos):
        part.name='Te'
    if part.pos[0] == 0 and part.pos[1] == 0 and part.pos[2] == .3359:
        part.name='Se'
    if part.pos[0] == .3359 and part.pos[1] == .3359 and part.pos[2] == .3359:
        part.name='Lv'
    #part.pos = np.matmul(np.array([[-2,1,1], [1,1,1], [0,1,-1]]),part.pos)
    part.pos = np.matmul(np.array([[0,1,0], [0,0,1], [1,0,0]]),part.pos)
    #if you want to do the above method ensure you normalize the rotmatrix
    print(part)


# In[ ]:




# In[ ]:

#small labeled cscl crystal 

import mbuild as mb
dim = 3
cscl_lengths = [.4123, .4123, .4123]
cscl_vectors = [[1,0,0], [0,1,0], [0,0,1]]
cscl_basis = {'Cs':[[0, 0, 0]], 'Cl':[[.5, .5, .5]]}
cscl_lattice = Lattice(cscl_lengths, dimension=dim,
                                lattice_vectors=cscl_vectors, basis_atoms=cscl_basis)
cs = mb.Compound(name='Cs')
cl = mb.Compound(name='Cl')
cscl_dict = {'Cs':cs, 'Cl':cl}
#cscl_crystal = cscl_lattice.populate(compound_dict=cscl_dict, x=3, y=3, z=3)
cscl_crystal = cscl_lattice.populate(compound_dict=cscl_dict, x=2, y=2, z=2)

cscl_crystal.visualize()


import numpy as np
for part in cscl_crystal:
    if np.array_equal([0,0,0], part.pos):
#         print(part.pos)
#         print(type(cscl_crystal))
#         print(part.name)
        part.name='Rb'
        
    if np.sum([1.0308,1.0308,1])<= np.sum(part.pos):
#         print(part.pos)
#         print(type(cscl_crystal))
#         print(part.name)
        part.name='F'
        
    if part.pos[0] == 0 and part.pos[1] == 0 and .1<= part.pos[2]<=.5:
#         print(part.pos)
#         print(type(cscl_crystal))
#         print(part.name)
        part.name='Fr'
    print(part)
        
print(cscl_crystal) 
print('issa test')
#cscl_crystal.visualize()


# In[ ]:

#rotating small labeled cscl

cscl_lattice.rotate_lattice(lat = cscl_crystal, new_view = [[-.5,.25,.25],[.25,.25,.25],[0,.25,-.25]], miller_directions = True)
print (type(cscl_lattice))


# In[ ]:

#this cannot work until the PR has been submitted

cscl_lattice.rotate_lattice(lat = cscl_crystal,
                            new_view = [[0,0,0],
                                        [0,0,1],
                                        [0.1,0.1,1]],
                            new_face = 'x')


# In[ ]:




# In[ ]:

#this is the example found here http://quantumwise.com/forum/index.php?topic=803.0
import mbuild as mb
#build
dim = 3
copper_lengths = [.352293, .352293, .352293]
copper_vectors = [[0,.5,.5], [.5,0,.5], [.5,.5,0]]
copper_basis = {'Cu':[[0,0,0]]}
copper_lattice_fcc = Lattice(copper_lengths, dimension=dim,
                                lattice_vectors=copper_vectors, basis_atoms=copper_basis)
cu = mb.Compound(name='Cu')
copper_dict = {'Cu':cu}
copper_crystal = copper_lattice_fcc.populate(compound_dict=copper_dict, x=2, y=2, z=2)


#rotate
print(type(copper_lattice_fcc))

print(type(copper_crystal))
copper_lattice_fcc.rotate_lattice(lat = copper_crystal, new_view = [[-2,1,1], [1,1,1], [0,1,-1]], miller_directions = True )
for part in copper_crystal:
    part.pos = part.pos/.352293
    print(part)

copper_crystal.visualize()


# In[ ]:




# In[ ]:




# In[ ]:

import hoomd
#import ex_render

hoomd.context.initialize('');
#system = hoomd.init.create_lattice()
uc = hoomd.lattice.sc(.3359, type_name = 'Po')
snap = uc.get_snapshot()
snap.replicate(2,2,2);
system = hoomd.init.read_snapshot(snap)
for part in snap.particles.position:
    print(part)
print(type(snap))
print(type(snap.particles))

#ucR = hoomd.lattice.sc(.3359, type_name = 'Po')
# uc = hoomd.lattice.unitcell(N = 1, a1 = [.3359,0,0],a2 = [0,.3359,0],a3 = [0,0,.3359],dimensions = 3,position = [[-.16795,-.16795,-.16795]])
# snap = uc.get_snapshot()
# snap.replicate(2,2,2)
# for part in snap.particles.position:
#     print(part)
    
# ucR = hoomd.lattice.unitcell(N = 1, a1 = [-2*.3359,.3359,.3359], a2 = [.3359,.3359,.3359], a3 = [0,.3359,-.3359],dimensions = 3, position = [[-.16795,-.16795,-.16795]])
# snap = ucR.get_snapshot()
# snap.replicate(2,2,2)
# for part in snap.particles.position:
#     print(part)


# In[ ]:



