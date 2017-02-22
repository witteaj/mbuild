
# coding: utf-8

# # Building a Simple Alkane

# The purpose of this tutorial is to demonstrate the construction of an alkane polymer and provide familiarity with many of the underlying functions in mBuild. Note that a robust polymer construction recipe already exists in mBuild, which will also be demonstrated at the end of the tutorial.

# ## Setting up the monomer

# The first step is to construct the basic repeat unit for the alkane, i.e., a $CH_2$ group, similar to the construction of the $CH_3$ monomer in the prior methane tutorial.  Rather than importing the coordinates from a pdb file, as in the previous example, we will instead explicitly define them in the class. Recall that distance units are nm in mBuild.

# In[2]:

import mbuild as mb

class CH2(mb.Compound):
    def __init__(self):
        super(CH2, self).__init__()
        self.add(mb.Particle(name='C', pos=[0,0,0]), label='C[$]')
        
        # Add hydrogens 
        self.add(mb.Particle(name='H', pos=[-0.109, 0, 0.0]), label='HC[$]')    
        self.add(mb.Particle(name='H', pos=[0.109, 0, 0.0]), label='HC[$]')
        
        # Add ports anchored to the carbon
        self.add(mb.Port(anchor=self[0]), label='up')
        self.add(mb.Port(anchor=self[0]), label='down')
        
        # Move the ports approximately half a C-C bond length away from the carbon
        mb.translate(self['up'], [0, -0.154/2, 0]) 
        mb.translate(self['down'], [0, 0.154/2, 0]) 

monomer = CH2()
monomer.visualize(show_ports=True)


# This configuration of the monomer is not a particularly realistic conformation. One could use this monomer to construct a polymer and then apply an energy minimization scheme, or, as we will demonstrate here, we can use mBuild's rotation commands to provide a more realistic starting point. 
# 
# Below, we use the same basic script, but now apply a rotation to the hydrogen atoms.  Since the hydrogens start 180&deg; apart and we know they should be ~109.5&deg; apart, each should be rotated half of the difference closer to each other around the y-axis. Note that the rotation angle is given in radians. Similarly, the ports should be rotated around the x-axis by the same amount so that atoms can be added in a realistic orientation. 

# In[5]:

import numpy as np
import mbuild as mb

class CH2(mb.Compound):
    def __init__(self):
        super(CH2, self).__init__()
        self.add(mb.Particle(name='C', pos=[0,0,0]), label='C[$]')
        self.add(mb.Particle(name='H', pos=[-0.109, 0, 0.0]), label='HC[$]')    
        self.add(mb.Particle(name='H', pos=[0.109, 0, 0.0]), label='HC[$]')
        theta = 0.5 * (180 - 109.5) * np.pi / 180
        mb.rotate(self['HC'][0], theta, around=[0, 1, 0])
        mb.rotate(self['HC'][1], -theta, around=[0, 1, 0])
        
        self.add(mb.Port(anchor=self[0]), label='up')
        mb.translate(self['up'], [0, -0.154/2, 0]) 
        mb.rotate(self['up'], theta, around=[1, 0, 0])
        self.add(mb.Port(anchor=self[0]), label='down')
        mb.translate(self['down'], [0, 0.154/2, 0]) 
        mb.rotate(self['down'], -theta, around=[1, 0, 0]) 

monomer = CH2()
monomer.visualize(show_ports=True)


# ## Defining the polymerization class

# With a basic monomer construct, we can now construct a polymer by connecting the ports together. Here, we first instantiate one instance of the CH2 class as ```1ast_monomer```, then use the clone function to make a copy. The ```force_overlap()``` function is used to connect the ```'up'``` port from ```current_monomer``` to the ```'down'``` port of ```last_mononer```.

# In[11]:

class AlkanePolymer(mb.Compound):
    def __init__(self):
        super(AlkanePolymer, self).__init__()
        last_monomer = CH2()
        self.add(last_monomer)
        for i in range(3):
            current_monomer = mb.clone(last_monomer)
            mb.force_overlap(move_this=current_monomer, 
                             from_positions=current_monomer['up'], 
                             to_positions=last_monomer['down'])
            self.add(current_monomer)
            last_monomer = current_monomer

polymer = AlkanePolymer()
polymer.visualize(show_ports=True)


# Visualization of this structure demonstrates a problem; the polymer curls up on itself.  This is a result of the fact that ports not only define the location in space, but also an orientation. This can be trivially fixed, by first rotating the port 180&deg; around the y-axis.  
# 
# We can also add a variable ```chain_length``` both to the for loop and ```init```  that will allow the length of the polymer to be adjusted when the class is instantiated.  

# In[13]:

import numpy as np
import mbuild as mb

class CH2(mb.Compound):
    def __init__(self):
        super(CH2, self).__init__()
        self.add(mb.Particle(name='C', pos=[0,0,0]), label='C[$]')
        self.add(mb.Particle(name='H', pos=[-0.109, 0, 0.0]), label='HC[$]')    
        self.add(mb.Particle(name='H', pos=[0.109, 0, 0.0]), label='HC[$]')
        theta = 0.5 * (180 - 109.5) * np.pi / 180
        mb.rotate(self['HC'][0], theta, around=[0, 1, 0])
        mb.rotate(self['HC'][1], -theta, around=[0, 1, 0])
        
        self.add(mb.Port(anchor=self[0]), label='up')
        mb.translate(self['up'], [0, -0.154/2, 0]) 
        mb.rotate(self['up'], theta, around=[1, 0, 0])
        self.add(mb.Port(anchor=self[0]), label='down')
        mb.translate(self['down'], [0, 0.154/2, 0])
        mb.rotate(self['down'], np.pi, [0, 1, 0])
        mb.rotate(self['down'], -theta, around=[1, 0, 0]) 


class AlkanePolymer(mb.Compound):
    def __init__(self, chain_length=1):
        super(AlkanePolymer, self).__init__()
        last_monomer = CH2()
        self.add(last_monomer)
        for i in range (chain_length-1):
            current_monomer = mb.clone(last_monomer)
    
            mb.force_overlap(move_this=current_monomer, 
                             from_positions=current_monomer['up'], 
                             to_positions=last_monomer['down'])
            self.add(current_monomer)
            last_monomer=current_monomer


# In[14]:

polymer = AlkanePolymer(chain_length=10)
polymer.visualize(show_ports=True)


# ## Using mBuild's Polymer Class

# ```mBuild``` provides a prebuilt class to perform this basic functionality. Since it is designed to be more general, it takes as an argument not just the chain length, but also the monomer and the port labels (e.g., 'up' and 'down', since these labels are user defined). 

# In[15]:

polymer = mb.Polymer(CH2(), 10, port_labels=('up', 'down'))
polymer.visualize()


# ## Building a System of Alkanes 

# A system of alkanes can be constructed by simply cloning the polymer constructed above and translating and/or rotating the alkanes in space. ```mBuild``` provides many routines that can be used to create different patterns, to which the polymers can be shifted.

# In[23]:

# create the polymer
polymer = mb.Polymer(CH2(), 10, port_labels=('up', 'down'))

# the pattern we generate puts points in the xy-plane, so we'll rotate the polymer
# so that it is oriented normal to the xy-plane
mb.rotate(polymer, np.pi/2, [1, 0, 0])

# define a compound to hold all the polymers
system = mb.Compound()

# create a pattern of points to fill a disk
# patterns are generated between 0 and 1, 
# and thus need to be scaled to provide appropriate spacing
pattern_disk = mb.DiskPattern(50)
pattern_disk.scale(5) 

# now clone the polymer and move it to the points in the pattern
for pos in pattern_disk:
    current_polymer = mb.clone(polymer)
    mb.translate(current_polymer, pos)
    system.add(current_polymer)
    
system.visualize()


# Other patterns can be used, e.g., the ```Grid3DPattern```.  We can also use the rotation commands to randomize the orientation. 

# In[27]:

import random

polymer = mb.Polymer(CH2(), 10, port_labels=('up', 'down'))
system = mb.Compound()
mb.rotate(polymer, np.pi/2, [1, 0, 0])

pattern_disk = mb.Grid3DPattern(5, 5, 5)
pattern_disk.scale(8.0)
    
for pos in pattern_disk:
    current_polymer = mb.clone(polymer)
    for around in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:  # rotate around x, y, and z
        mb.rotate(current_polymer, random.uniform(0, np.pi), around)
    mb.translate(current_polymer, pos)
    system.add(current_polymer)

system.visualize()


# ```mBuild``` also provides an interface to ```PACKMOL```, allowing the creation of a randomized configuration.

# In[29]:

polymer = mb.Polymer(CH2(), 5, port_labels=('up', 'down'))
system = mb.fill_box(polymer, n_compounds=100, overlap=1.5, box=[10,10,10]) 
system.visualize()


# ## Variations

# Rather than a linear chain, the ```Polymer``` class we wrote can be easily changed such that small perturbations are given to each port. To avoid accumulation of deviations from the equilibrium angle, we will clone an unperturbed monomer each time (i.e., ```monomer_proto```) before applying a random variation.  
# 
# We also define a variable ```delta```, which will control the maximum amount of perturbation. Note that large values of ```delta``` may result in the chain overlapping itself, as ```mBuild``` does not currently include routines to exclude such overlaps.

# In[33]:

import mbuild as mb

import random

class AlkanePolymer(mb.Compound):
    def __init__(self, chain_length=1, delta=0):
        super(AlkanePolymer, self).__init__()
        monomer_proto = CH2()
        last_monomer = CH2()
        mb.rotate(last_monomer['down'], random.uniform(-delta,delta), [1, 0, 0])
        mb.rotate(last_monomer['down'], random.uniform(-delta,delta), [0, 1, 0])
        self.add(last_monomer)
        for i in range(chain_length-1):
            current_monomer = mb.clone(monomer_proto)
            mb.rotate(current_monomer['down'], random.uniform(-delta,delta), [1, 0, 0])
            mb.rotate(current_monomer['down'], random.uniform(-delta,delta), [0, 1, 0])
            mb.force_overlap(move_this=current_monomer, 
                             from_positions=current_monomer['up'], 
                             to_positions=last_monomer['down'])
            self.add(current_monomer)
            last_monomer=current_monomer

polymer = AlkanePolymer(chain_length = 200, delta=0.4)
polymer.visualize()

