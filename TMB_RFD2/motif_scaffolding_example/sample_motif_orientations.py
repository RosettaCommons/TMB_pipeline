#! /software/containers/pyrosetta.sif
''' This script will take a a pdb file as input. Then move all atoms to the origin. Then randomly pick an axis and rotate by a random angle and translate in space by a small distance close to the origin'''

import os
import argparse
import numpy as np
import random

def parse_args():
    ''' Parse command line arguments '''
    parser = argparse.ArgumentParser(description='This script will take a small molecule file as one input and a pdb file containing a pore-forming beta-barrel protein and will output a pdb file containing the small molecule in the pore of the protein. Depending on the desired number of output pdb files, the small molecule will be randomly docked and each docked pose will be output as a separate pdb file. A docking score based on Calpha and small molecule distances will be calculated for each docked pose with a minimum approach distance of 5.0 Angstroms between any atom of the small molecule and any Calpha atom of the protein.')
    parser.add_argument('-m', '--motif', type=str, required=False, help='Protein file in pdb format', default='./smallMoleculeDock_test/sucrose_TMBsol93.pdb')
    parser.add_argument('-n', '--num_docked', type=int, required=False, help='Number of docked poses to generate', default=1)
    parser.add_argument('-o', '--output_prefix', type=str, required=False, help='Prefix for output pdb files', default='./sampled_pose')
    parser.add_argument('-r', '--random_seed', type=int, required=False, help='Random seed for docking', default=None)
    args = parser.parse_args()
    return args

def read_pdb(pdb_file):
    ''' Read a pdb file and return a list of lines '''
    with open(pdb_file, 'r') as f:
        pdb_lines = f.readlines()
    return pdb_lines

def get_pdb_info(pdb_lines):
    ''' Get information from pdb file '''
    protein_lines = []
    small_molecule_lines = []
    conect_lines = []
    for line in pdb_lines:
        if line.startswith('ATOM'):
            protein_lines.append(line)
        elif line.startswith('HETATM'):
            small_molecule_lines.append(line)
        elif line.startswith('CONECT'):
            conect_lines.append(line)
    return protein_lines, small_molecule_lines, conect_lines

def get_protein_info(protein_lines):
    ''' Get protein information from pdb file '''
    protein_info = []
    for line in protein_lines:
        line = line.strip()
        line = line.split()
        protein_info.append(line)
    return protein_info


def get_all_coords(protein_info):
    
    ''' Get all coordinates from pdb file '''
    protein_coords = []
    for line in protein_info:
        if len(line[4]) > 1:
            protein_coords.append(np.float64(line[5:8]))
        else:
            protein_coords.append(np.float64(line[6:9]))
    return protein_coords


def get_small_molecule_center(small_molecule_coords):
    ''' Get center of small molecule '''
    small_molecule_center = []
    for n in range(len(small_molecule_coords)):
        for i in range(3):
            small_molecule_center.append(float(small_molecule_coords[n][i]))
    small_molecule_center = np.array(small_molecule_center)
    small_molecule_center = np.reshape(small_molecule_center, (len(small_molecule_coords), 3))
    small_molecule_center = np.mean(small_molecule_center, axis=0)
    return small_molecule_center

def get_small_molecule_centered_coords(small_molecule_coords, small_molecule_center):
    ''' Get small molecule coordinates centered on the origin '''
    small_molecule_centered_coords = []
    for n in range(len(small_molecule_coords)):
        for i in range(3):
            small_molecule_centered_coords.append(float(small_molecule_coords[n][i]) - small_molecule_center[i])
    small_molecule_centered_coords = np.array(small_molecule_centered_coords)
    small_molecule_centered_coords = np.reshape(small_molecule_centered_coords, (len(small_molecule_coords), 3))
    return small_molecule_centered_coords

def get_small_molecule_major_axis(small_molecule_centered_coords):
    ''' Get principal axes of small molecule '''
    mat = np.linalg.eig(np.cov(small_molecule_centered_coords.T))
    eigmax = np.argmax(mat[0])
    small_molecule_major_axis = mat[1][eigmax]
    return small_molecule_major_axis


def get_small_molecule_rot_matrix(small_molecule_major_axis):
    ''' Rigid body rotate small molecule to align major axis with z-axis. First find perpendicular axis to both
    small_moelcule_major_axis and z-axis. Then find rotation matrix to rotate small molecule_major_axis to z-axis. '''
    dotprod = np.dot(small_molecule_major_axis, np.array([0, 0, 1]))
    crossprod = np.cross(small_molecule_major_axis, np.array([0, 0, 1]))
    skewsym = np.array([[0, -crossprod[2], crossprod[1]], [crossprod[2], 0, -crossprod[0]], [-crossprod[1], crossprod[0], 0]])
    small_molecule_rot_matrix = np.identity(3) + skewsym + np.dot(skewsym, skewsym)*(1.0/(1.0 + dotprod))
    return small_molecule_rot_matrix



def get_small_molecule_aligned_coords(small_molecule_centered_coords, small_molecule_rot_matrix):
    ''' Get small molecule coordinates aligned to z-axis '''
    small_molecule_aligned_coords = []
    for n in range(len(small_molecule_centered_coords)):
        small_molecule_aligned_coords.append(np.dot(small_molecule_rot_matrix, small_molecule_centered_coords[n]))
    small_molecule_aligned_coords = np.array(small_molecule_aligned_coords)
    return small_molecule_aligned_coords


def get_random_translation(protein_height):
    ''' Get random translation for small molecule '''
    random_translation = np.random.uniform(-protein_height/3.0, protein_height/3.0)
    return random_translation


def get_rotation_matrix(random_rotation):
    ''' Get rotation matrix for small molecule '''
    rotation_matrix = np.array([[np.cos(random_rotation), -np.sin(random_rotation), 0], [np.sin(random_rotation), np.cos(random_rotation), 0], [0, 0, 1]])
    return rotation_matrix

def get_translation_matrix(random_translation):
    ''' Get transform matrix for small molecule '''
    translation_matrix = np.array([0, 0, random_translation])
    return translation_matrix

def get_transformed_coords(small_molecule_aligned_coords, translation_matrix, rotation_matrix):
    ''' Get transformed coordinates for small molecule '''
    transformed_coords = []
    for n in range(len(small_molecule_aligned_coords)):
        transformed_coords.append(np.dot(rotation_matrix, small_molecule_aligned_coords[n]) + translation_matrix)
    transformed_coords = np.array(transformed_coords)
    return transformed_coords
    

def calculate_distances(protein_coords, transformed_coords):
    ''' Calculate distances between protein Calpha atoms and small molecule atoms '''
    distances = []
    for n in range(len(protein_coords)):
        for i in range(len(transformed_coords)):
            distances.append(np.linalg.norm(np.array(protein_coords[n]) - np.array(transformed_coords[i])))
    distances = np.array(distances)
    distances = np.reshape(distances, (len(protein_coords), len(transformed_coords)))
    return distances

def get_min_distance(distances):
    ''' Get minimum distance between protein Calpha atoms and small molecule atoms '''
    min_distance = np.min(distances)
    return min_distance


'''The following functions will laterally move the molecule at a given Z-axis position with distances chosen from a uniform distribution between -4 and 4 Angstroms.
They will also rotate the molecule around any randomly chosen axis by 180 degrees. After each of these operations, a clash check will be performed between all atoms of the
molecule and nearby Calpha atoms of the protein. If a clash is detected, the molecule will be moved back to its original position and the operation will be repeated until one dock
clears the filter.'''

def get_rot_mat_arbitrary_axis(random_rotation):
    '''Choose an arbitrary axis for rotation. Then use the input angle to get the rotation matrix.'''
    arbitrary_axis = np.random.rand(3)
    arbitrary_axis = arbitrary_axis/np.linalg.norm(arbitrary_axis)
    rotation_matrix = np.array([[np.cos(random_rotation) + arbitrary_axis[0]**2*(1-np.cos(random_rotation)), arbitrary_axis[0]*arbitrary_axis[1]*(1-np.cos(random_rotation)) - arbitrary_axis[2]*np.sin(random_rotation), arbitrary_axis[0]*arbitrary_axis[2]*(1-np.cos(random_rotation)) + arbitrary_axis[1]*np.sin(random_rotation)], [arbitrary_axis[1]*arbitrary_axis[0]*(1-np.cos(random_rotation)) + arbitrary_axis[2]*np.sin(random_rotation), np.cos(random_rotation) + arbitrary_axis[1]**2*(1-np.cos(random_rotation)), arbitrary_axis[1]*arbitrary_axis[2]*(1-np.cos(random_rotation)) - arbitrary_axis[0]*np.sin(random_rotation)], [arbitrary_axis[2]*arbitrary_axis[0]*(1-np.cos(random_rotation)) - arbitrary_axis[1]*np.sin(random_rotation), arbitrary_axis[2]*arbitrary_axis[1]*(1-np.cos(random_rotation)) + arbitrary_axis[0]*np.sin(random_rotation), np.cos(random_rotation) + arbitrary_axis[2]**2*(1-np.cos(random_rotation))]])
    return rotation_matrix

def get_xy_trans_mat():
    '''Get translation matrix for x and y directions'''
    translation_matrix = np.array([np.random.uniform(-4.0, 4.0), np.random.uniform(-4.0, 4.0), 0])
    return translation_matrix


def random_planar_dock_sampling(transformed_coords):
    ''' Randomly dock small molecule in pore '''
    translation_matrix = get_xy_trans_mat()
    rotation_matrix = np.array([[np.cos(0.0), -np.sin(0.0), 0], [np.sin(0.0), np.cos(0.0), 0], [0, 0, 1]])
    new_coords = get_transformed_coords(transformed_coords, translation_matrix, rotation_matrix)
    return new_coords


def get_docked_pose(protein_lines, transformed_coords, nonsm):
    ''' Write the transformed coordinates in a pdb file format and append each line of the new pdb file to the docked_pose list '''
    docked_pose = []
    for i, line in enumerate(protein_lines):
        if "ATOM" in line:
            # Make sure the string of converted coordinates is 8 characters long
            line = line[:30] + "{:8.3f}".format(transformed_coords[i][0]) + "{:8.3f}".format(transformed_coords[i][1]) + "{:8.3f}".format(transformed_coords[i][2]) + line[54:]
        docked_pose.append(line)
            
    docked_pose.append('END\n')
    return docked_pose

def write_pdb(docked_pose, output_prefix, n):
    ''' Write docked pose to pdb file '''
    with open(output_prefix + '_' + 'des'+ str(n) + '.pdb', 'w') as f:
        for line in docked_pose:
            f.write(line)

''' The main function will iterate over the number of desired docked poses and will output each docked pose to a separate pdb file. '''

def main():
    nonsm = False
    args = parse_args()
    seed = args.random_seed
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    pref = args.output_prefix.split("/")[-1]
    odir = "/".join(args.output_prefix.split("/")[:-1])
    if not os.path.exists(odir):
        os.mkdir(odir)
    opref = odir + "/" + pref
    
    pdb_lines = read_pdb(args.motif)
    if args.motif.endswith(".pdb") == False:
        print("Please provide a pdb file")
        exit(1)
    protein_lines, sml, cl = get_pdb_info(pdb_lines)

    protein_info = get_protein_info(protein_lines)
    protein_coords = get_all_coords(protein_info)
    small_molecule_center = get_small_molecule_center(protein_coords)
    small_molecule_centered_coords = get_small_molecule_centered_coords(protein_coords, small_molecule_center)
    small_molecule_major_axis = get_small_molecule_major_axis(small_molecule_centered_coords)
    small_molecule_rot_matrix = get_small_molecule_rot_matrix(small_molecule_major_axis)
    small_molecule_aligned_coords = get_small_molecule_aligned_coords(small_molecule_centered_coords, small_molecule_rot_matrix)
    count = 1
    for n in range(args.num_docked*10):
        random_translation = get_random_translation(np.max(small_molecule_aligned_coords[:, 2]) - np.min(small_molecule_aligned_coords[:, 2]))
        random_rotation = np.random.uniform(0.0, 2.0*np.pi)
        rotation_matrix = get_rot_mat_arbitrary_axis(random_rotation)
        translation_matrix = get_translation_matrix(random_translation)
        transformed_coords = get_transformed_coords(small_molecule_aligned_coords, translation_matrix, rotation_matrix)
        distances = calculate_distances(small_molecule_aligned_coords, transformed_coords)
        min_distance = get_min_distance(distances)
        if min_distance < 0.4:
            xy_perturbed_coords = random_planar_dock_sampling(transformed_coords)
            docked_pose = get_docked_pose(protein_lines, xy_perturbed_coords, nonsm)
            write_pdb(docked_pose, opref + "_" + args.motif.split("/")[-1][:-4], count)
            count += 1
        if count > args.num_docked:
            break

if __name__ == '__main__':
    main()







