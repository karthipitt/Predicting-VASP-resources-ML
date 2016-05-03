#!/usr/bin/env python
import pandas as pd
import numpy as np
import json,pickle
import itertools
import re
from subprocess import Popen, PIPE
import os

df = pd.read_csv('newconvertcsv.csv')

# Cleaning the dataset and selecting required columns
'''
NBANDS 
Number of atoms
Types of Atoms
Volume
Number of relaxed atoms
Type of calculation (Bulk/Surface/Ads)
ENCUT
k-Points
Precision
XC
IBRION
ISYM
ISIF
Number of valence electrons
Elapsed Time
Memory
'''

raw_features = ['incar/nbands','incar/isif','incar/ibrion','incar/nsw','incar/ismear','incar/sigma','incar/ispin','incar/algo','incar/gga','results/type','incar/encut','input/kpts/0','input/kpts/1','input/kpts/2','incar/prec','input/xc','data/volume','results/elapsed-time','results/memory-used']
calc_features = ['natoms','type_of_atoms','n_val_electrons','n_relaxed_atoms']

atom_cols = [col for col in df if 'atoms/symbols' in col]
atoms_df = df[atom_cols] 
tot_atoms = atoms_df.count(axis=1) #Total Number of atoms in each calculation
tot_atoms.name = 'total_atoms'

def cunique(row):
    rmnan = [g for g in row if not pd.isnull(g)] 
    return [(g[0], len(list(g[1]))) for g in itertools.groupby(rmnan) ]

atom_list = atoms_df.apply(func=cunique,axis=1) 

# Returns a dictionary with number of valence electrons of each atom
def nval_ele():
    val_e_dict = {}
    pp = os.environ['VASP_PP_PATH']
    pattern = re.compile("potcar/[0-9]/0")
    type_atom_col = [col for col in df if pattern.match(col)]
    unique_atoms =  pd.unique(df[type_atom_col].values.ravel())
    for i,atoms in enumerate(unique_atoms):
        if not pd.isnull(atoms):
            p = Popen('grep "ZVAL" {0}/potpaw_PBE/{1}/POTCAR '.format(pp,atoms), stdin=PIPE, stdout=PIPE, stderr=PIPE,shell=True)
            out, err = (p.communicate())
            if out and not err:
                val_e_dict[atoms]=float(out.split()[5])
            else:
                print err
    return val_e_dict

val_e_dict = nval_ele()    

# Calculate number of valence electrons and append it to dataframe
def calc_val_electrons(row):
    val_e = 0
    for elem in row:
        at = elem[0]
        atc = elem[1]
        val_e = val_e + atc * val_e_dict[at]
    return val_e
atom_val_e = atom_list.apply(func=calc_val_electrons)
atom_val_e.name='total_val_ele'

## Number of constrained atoms
#f = open('vasp-ml.json')
#d = json.loads(f.read())
#
#cons_atoms = []
#for i,x in enumerate(d):
#    print tot_atoms[i],
#    if 'atoms.constraints' in x['metadata'].keys():
#        constraints = pickle.loads(x['metadata']['atoms.constraints'].encode('utf-8'))
#        cons = constraints[0].index
#        if isinstance(cons[2],int):
#            cons_atoms.append(len(cons))
#        else:
#            cons_atoms.append(sum(cons))
#    else:
#        cons_atoms.append(0)

#print df['metadata/atoms.constraints'].apply(lambda x: pickle.loads(x.encode('utf-8')))


# Handling missing values of each feature
all_data = pd.concat([df[raw_features],tot_atoms,atom_val_e],axis=1)
fill_defaults = {'incar/isif':2,'incar/ibrion':0,'incar/nsw':0,'incar/ismear':1,'incar/sigma':0.2,'incar/ispin':1,'incar/algo':'N','incar/gga':'PE','incar/prec':'Normal'}

for key in fill_defaults:
    all_data[key] = all_data[key].fillna(fill_defaults[key])

# Rename columns to make labels visible while plotting
for col in all_data:
    new_col_name = col.split('/')[-1]
    all_data = all_data.rename(columns={col:new_col_name})

all_data = all_data.rename(columns={'0':'kp0'})
all_data = all_data.rename(columns={'1':'kp1'})
all_data = all_data.rename(columns={'2':'kp2'})

all_data['elapsed-time'] = all_data['elapsed-time']/60
   
#all_data.to_csv('clean_data.csv',index=False) 
