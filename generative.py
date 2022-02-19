
#
#   blender 3.0.1.
#

import bpy
from bpy import data
from bpy import context
import bmesh
from bmesh import geometry
import mathutils
from mathutils.bvhtree import BVHTree

import numpy as np
from numpy.random import default_rng

import math

# triangulate using bmesh.
def triangulate(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()  # free and prevent further access

def select_activate_only(objects=[]):
    for obj in bpy.data.objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = None 
    for obj in objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

# https://blender.stackexchange.com/questions/220072/check-using-name-if-a-collection-exists-in-blend-is-linked-to-scene
def create_collection_if_not_exists(collection_name):
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection) #Creates a new collection

# https://graphics.pixar.com/library/OrthonormalB/paper.pdf
def pixar_onb(n):
    t = mathutils.Vector((0,0,0))
    b = mathutils.Vector((0,0,0))
    if(n[2] < 0.0):
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        t = mathutils.Vector((1.0 - n[0] * n[0] * a, -b, n[0]))
        b = mathutils.Vector((b, n[1] * n[1] * a - 1.0, -n[1]))
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        t = mathutils.Vector((1.0 - n[0] * n[0] * a, b, -n[0]))
        b = mathutils.Vector((b, 1 - n[1] * n[1] * a, -n[1]))
    return t, b

def create_instance(base_obj,
                    translate=mathutils.Vector((0,0,0)), 
                    scale=1.0,
                    rotate=("Z", 0.0),
                    basis=mathutils.Matrix.Identity(4),
                    tbn=mathutils.Matrix.Identity(4),
                    collection_name=None):
    # Create instance.
    base_obj_mesh = base_obj.data
    inst_obj = bpy.data.objects.new(base_obj.name+"_inst", base_obj_mesh)
    # Perform translation, rotation, scaling and moving to target coord system for instance.
    mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
    mat_trans = mathutils.Matrix.Translation(translate)
    mat_sca = mathutils.Matrix.Scale(scale, 4) # TODO: figure out how to scale in given vector direction
    # TODO: If I am using `tbn` as basis then it sould go last, If I use `matrix_basis` as basis then it should go first.
    # `tbn` matrix is usually constructed for samples on base geometry using triangle normal. Therefore, it only contains
    # information about rotation.
    inst_obj.matrix_basis = basis @ mat_trans @ mat_rot @ mat_sca @ tbn  # TODO: is matrix_basis correct to be used for this?
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(inst_obj)
    else:
        bpy.data.collections[collection_name].objects.link(inst_obj)
    return inst_obj

# 
# Description:
# Create a cube using bmesh in center of world coordinate system.
# Apply translation, rotation and scaling using translate_vector, scale, rotation_axis_angle
# Apply rotation and translation to final coordinate system using basis
#
# Parameters:
# size - overall cube size
# translate - translation for given vector
# scale - scaling cube in specific vector direction
# rotate - rotate around given vector
# triangulate - triangulate quads into triangles
# basis - rotation and translation matrix defining target coordinate system
#
def create_cube(translate=mathutils.Vector((0,0,0)), 
                scale=1.0,
                rotate=("Z", 0.0),
                triangulate=False,
                basis=mathutils.Matrix.Identity(4), # rotation and translation defining target (final) coordinate system
                tbn=mathutils.Matrix.Identity(4),
                collection_name=None): # collection destination
    # Create unit cube in center.
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
    if triangulate:
        bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="BEAUTY", ngon_method="BEAUTY")
    object_mesh = bpy.data.meshes.new("cube_mesh")
    bm.to_mesh(object_mesh)
    bm.free()
    obj = bpy.data.objects.new("cube_obj", object_mesh)
    # Add transformation.
    mat_rot = mathutils.Matrix.Rotation(math.radians(rotate[1]), 4, rotate[0])
    mat_trans = mathutils.Matrix.Translation(translate)
    mat_sca = mathutils.Matrix.Scale(scale, 4)
    # TODO: If I am using tbn as basis then it sould go last, If I use matrix_basis as basis then it should go first?
    obj.matrix_basis = basis @ mat_trans @ mat_rot @ mat_sca @ tbn # TODO: is matrix_basis correct to be used? matrix_local vs matrix_parent_inverse vs matrix_world
    # Store to collection.
    if collection_name == None:
        bpy.context.collection.objects.link(obj)
    else:
        bpy.data.collections[collection_name].objects.link(obj)

    return obj

#
# barycentric sampling of triangulated mesh using bmesh.
# It seems that bmesh doesn't have any additional info on vertices (e.g. weight, color, etc.)
# n_samples - number of samples per mesh triangle.
#
def barycentric_sampling(base_obj, n_samples):
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(base_obj.data)   # fill it in from a Mesh
    samples = [] # (p, N) 
    for face in bm.faces:
        triangle_points = []
        for loop in face.loops: # todo: triangle.faces?
            triangle_points.append(loop.vert.co)
        for i in range(n_samples):
            x = mathutils.noise.random()
            y = mathutils.noise.random()
            z = mathutils.noise.random()
            s = x + y + z
            p = (x / s) * triangle_points[0] + (y / s) * triangle_points[1] + (z / s) * triangle_points[2] # NOTE: random barycentric coord
            samples.append((mathutils.Vector(p), mathutils.Vector(face.normal))) # NOTE: new vector is created so that new object rather than reference is stored
    bm.free()  # free and prevent further access
    return samples

def jitter_around_triangle_sample(triangle_vertices, base_sample, n_epsilon, epsilon, triangle_area=0.0):
    jitter_samples = []
    while True:
        if len(jitter_samples) >= n_epsilon:
            break
        u = mathutils.noise.random()
        v = mathutils.noise.random()
        w = mathutils.noise.random()
        s = u + v + w
        un = (u / s)
        vn = (v / s)
        wn = (w / s)
        p = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
        base_jitter = base_sample - p
        if base_jitter.length < epsilon * triangle_area: # TODO: enhance useage of triangle area!
            # NOTE: samples are defined by their barycentric factors.
            jitter_samples.append(list((un,vn,wn)))
    return jitter_samples

#
# barycentric sampling of triangulated mesh using object.data (mesh)
# object.data (mesh) has extensive info on vertices (e.g. weight, color, etc.)
#
# `n_epsilon` - number of additional samples per sample in `epsilon` neighbourhood.
# `epsilon` - depends on triangle size and later size of growth elements.
#
def vertex_weighted_barycentric_sampling_jitter(base_obj, n_samples, n_epsilon=0, epsilon=0.0):
    samples = [] # (p, N, w, tbn)
    for polygon in base_obj.data.polygons: # must be triangulated mesh!
        triangle_vertices = []
        triangle_vertex_weights = []
        for v_idx in polygon.vertices:
            v = base_obj.data.vertices[v_idx]
            triangle_vertices.append(v.co)
            if len(v.groups) < 1:
                triangle_vertex_weights.append(0.0)
            else:
                triangle_vertex_weights.append(v.groups[0].weight) # TODO: only one group? Investigate! float in [0, 1], default 0.0
        for i in range(n_samples):
            # Find sample using barycentric sampling.
            a = mathutils.noise.random()
            b = mathutils.noise.random()
            c = mathutils.noise.random()
            s = a + b + c
            un = (a / s)
            vn = (b / s)
            wn = (c / s)
            p_original = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
            # NOTE: we later use w_original to calculate w of other neighbourhood samples. This way neigh samples are not 
            # discarded due to mesh weight in that point. Other possibility is to calculate w for each neighbourhood using
            # interpolation.
            w_original = un * triangle_vertex_weights[0] + vn * triangle_vertex_weights[1] + wn * triangle_vertex_weights[2] # interpolate weight
            if w_original > 0.01: # probably an error otherwise
                # Find more samples in neighbourhood of the found one.
                all_samples = jitter_around_triangle_sample(triangle_vertices, p_original, n_epsilon, epsilon, polygon.area)
                all_samples.append(list((un,vn,wn)))
                for sample in all_samples:
                    p = sample[0] * triangle_vertices[0] + sample[1] * triangle_vertices[1] + sample[2] * triangle_vertices[2]
                    #w = sample[0] * triangle_vertex_weights[0] + sample[1] * triangle_vertex_weights[1] + sample[2] * triangle_vertex_weights[2] # interpolate weight
                    w = w_original * (1.0 - 0.5) * mathutils.noise.random() + 0.5
                    n = polygon.normal # TODO: vertex normals?
                    # Calc tangent. NOTE: use most distant point from barycentric coord to evade problems with 0
                    t = mathutils.Vector(triangle_vertices[0] - p)
                    t = t.normalized()
                    bt = n.cross(t)
                    bt = bt.normalized()
                    tbn = mathutils.Matrix((t, bt, n)) # NOTE: using pixar_onb()?
                    tbn = tbn.transposed() # TODO: why transposing?
                    tbn.resize_4x4()
                    samples.append((p,n,w,tbn))
    return samples 

# base_object prev object on which we build.
def jitter_offset(N, jitter_strength):
    # Use normal for finding plane on which jitter can take place.
    t, b = pixar_onb(N)
    u = (2*mathutils.noise.random()-1) * jitter_strength
    v = (2*mathutils.noise.random()-1) * jitter_strength
    return u * t + v * b 

# sample - (p, N, w, tbn)
# base_elements_collection_name - blender collection of object elements to be replicated
# max_elements_size - determines the scaling of largest base object dimension for using it as elements size.
# jitter_strength - determines offset scaling
def grow_from_sample(base_object,
                    sample, 
                    base_elements_collection_name,
                    max_grow_steps,  
                    max_elements_size, 
                    jitter_strength=0.0,
                    rotation_min_max=(0.0, 60.0)):
    base_element_names = []
    for base_element in bpy.data.collections[base_elements_collection_name].all_objects:
        base_element_names.append(base_element.name)
    p = sample[0]
    N = sample[1]
    w = sample[2]
    tbn = sample[3]
    # Use object max dimensions and weight in that point for setting initial size.
    size_curr = max(base_object.dimensions) * max_elements_size * w
    n_steps = math.ceil(max_grow_steps * w) # weight in sample determines the actual number of steps
    rng = default_rng()
    rand_base_element_indices = rng.integers(len(base_element_names), size=n_steps)
    rotation_curr = 0.0 # always add to rotation - add to last
    grow_obj_names = []
    for i in range(n_steps):
        # NOTE: current step size depends on size_curr: current element should at least touch previous element.
        curr_step_size = size_curr * ((1.0-0.7)*mathutils.noise.random()+0.7) # Step size is rand from [0, (size_curr)]
        # NOTE: Jittering strenght depends on current element size.
        jitter_vector = jitter_offset(N, size_curr * jitter_strength) 
        p = p + N * curr_step_size + jitter_vector
        if mathutils.noise.random() < 0.0:
            rotation_curr = rotation_curr + (rotation_min_max[1] - rotation_min_max[0]) * mathutils.noise.random() + rotation_min_max[0]
        curr_element = bpy.data.collections[base_elements_collection_name].all_objects[base_element_names[rand_base_element_indices[i]]]
        inst = create_instance(base_obj=curr_element, translate=mathutils.Vector(p), scale=size_curr, rotate=(N, rotation_curr), basis=mathutils.Matrix.Identity(4), tbn=tbn)
        grow_obj_names.append(inst.name)
        # Generally, I want scale to be smaller and smaller. But not necessary too small in the end!
        if mathutils.noise.random() < 0.4:
            size_curr = size_curr * 0.90
        else:
            size_curr = size_curr * 1.1
    return grow_obj_names

# orbiter_size = [0,1] of base_object dimensions
# orbiter_distance = [0,1] of center object dimensions
def create_orbiters(base_object,
                    n_samples_per_triangle, 
                    orbiter_distance, 
                    orbiter_size, 
                    acceptance_ratio):
    # Sample triangles of given base mesh.
    surface_orbiters = barycentric_sampling(base_object, n_samples_per_triangle) # (p,N)
    n_orbiters = len(surface_orbiters)
    # Create orbiter objects.
    orbiter_objects = [] 
    for i in range(n_orbiters):
        if mathutils.noise.random() > acceptance_ratio:
            continue
        surface_orbiter = surface_orbiters[i]
        p = surface_orbiter[0]
        N = surface_orbiter[1]
        orbiter_pos = p + N * mathutils.noise.random() * orbiter_distance * max(base_object.dimensions) # TODO: enhance dist?
        orbiter_scale = mathutils.noise.random() * orbiter_size * max(base_object.dimensions) # TODO: enhance size?
        orbiter_object = create_cube(translate=mathutils.Vector(orbiter_pos), scale=orbiter_scale, rotate=("Z", 0.0), triangulate=False, basis=base_object.matrix_basis, tbn=mathutils.Matrix.Identity(4))
        orbiter_objects.append(orbiter_object)
    # Join orbiter objects.
    orbiter_objects.append(base_object)
    select_activate_only(orbiter_objects)
    bpy.ops.object.join() # name of a joined object is same as the name as the last object in list

def create_orbited_base_objects(destination_collection_name,
                                n_base_objects,
                                n_samples_per_triangle, 
                                orbiter_distance, 
                                orbiter_size, 
                                acceptance_ratio):
    for i in range(n_base_objects):
        # Create unit cube and place it in destination collection.
        cube = create_cube(translate=mathutils.Vector((0,0,0)), 
                scale=1.0,
                rotate=("Z", 0.0),
                triangulate=True,
                basis=mathutils.Matrix.Identity(4),
                tbn=mathutils.Matrix.Identity(4),
                collection_name=destination_collection_name)
        # Create orbiters for that cube.
        create_orbiters(base_object=cube, 
                        n_samples_per_triangle=n_samples_per_triangle, 
                        orbiter_distance=orbiter_distance,  
                        orbiter_size=orbiter_size, 
                        acceptance_ratio=acceptance_ratio)

def boolean_difference(with_object):
    bpy.ops.object.modifier_add(type="BOOLEAN")
    bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
    bpy.context.object.modifiers["Boolean"].object = with_object
    bpy.context.object.modifiers["Boolean"].solver = "EXACT" # TODO: approx?
    bpy.context.object.modifiers["Boolean"].use_self = False
    bpy.ops.object.modifier_apply(modifier="Boolean")

# max_cavity_size [0,1] of base object size
# sample_acceptance_ratio [0, 1]
def create_cavities(base_obj, samples_per_triangle, sample_acceptance_ratio, max_cavity_size):
    cavity_positions = barycentric_sampling(base_obj, samples_per_triangle)
    for cavity_position in cavity_positions:
        if mathutils.noise.random() < sample_acceptance_ratio:
            p = cavity_position[0]
            N = cavity_position[1]
            #loc, rot, sca = base_obj.matrix_basis.decompose()
            #basis = mathutils.Matrix.LocRotScale(loc, rot, None)
            cavity_scale = max_cavity_size * mathutils.noise.random() * base_obj.dimensions[0]
            cavity_negative = create_cube(translate=mathutils.Vector(p), 
                                            scale=cavity_scale, 
                                            rotate=("Z", 0.0), 
                                            triangulate=False, 
                                            basis=base_obj.matrix_basis, 
                                            tbn=mathutils.Matrix.Identity(4))
            select_activate_only([base_obj])
            boolean_difference(cavity_negative)
            bpy.data.objects.remove(cavity_negative, do_unlink=True)

def create_cavitated_base_objects(collection_name, 
                                    n_base_objects,
                                    n_cavities_per_triangle, 
                                    cavities_per_triangle_acceptance, 
                                    max_cavity_size):
    for i in range(n_base_objects):
        cube = create_cube(translate=mathutils.Vector((0,0,0)), 
                scale=1.0,
                rotate=("Z", 0.0),
                triangulate=True,
                basis=mathutils.Matrix.Identity(4),
                tbn=mathutils.Matrix.Identity(4),
                collection_name=collection_name)
        create_cavities(base_obj=cube, 
                        samples_per_triangle=n_cavities_per_triangle, 
                        sample_acceptance_ratio=cavities_per_triangle_acceptance, 
                        max_cavity_size=max_cavity_size)
        #triangulate(cube)

def main():
    # Create samples on selected object.
    base_obj = bpy.context.selected_objects[0]
    triangulate(base_obj)
    samples = vertex_weighted_barycentric_sampling_jitter(base_obj=base_obj, 
                                                            n_samples=1, 
                                                            n_epsilon=2, 
                                                            epsilon=0.2)
    # Pre-comp: Fetch existing or generate base elements.
    base_elements_collection_name = "base_elements"
    create_collection_if_not_exists(base_elements_collection_name)
    if len(bpy.data.collections[base_elements_collection_name].all_objects) == 0:
        # Cavitated base elements.
        create_cavitated_base_objects(collection_name=base_elements_collection_name, 
                                        n_base_objects=15, 
                                        n_cavities_per_triangle=2, 
                                        cavities_per_triangle_acceptance=1.0, 
                                        max_cavity_size=0.5)
        # Extruded (via orbiting) base elements.
        create_orbited_base_objects(destination_collection_name=base_elements_collection_name,
                                    n_base_objects=15,
                                    n_samples_per_triangle=1, 
                                    orbiter_distance=0.0, 
                                    orbiter_size=0.4, 
                                    acceptance_ratio=1.0)
    # Grow on samples using precomp objects.
    grow_cluster_names = []   
    for i in range(len(samples)):
        grow_cluster_name = grow_from_sample(base_object=base_obj,
                                                sample=samples[i], 
                                                base_elements_collection_name=base_elements_collection_name, 
                                                jitter_strength=0.2, 
                                                max_grow_steps=10, 
                                                max_elements_size=0.04, 
                                                rotation_min_max=(0.0, 0.0))
        grow_cluster_names.append(grow_cluster_name)

if __name__ == "__main__":
    main()
