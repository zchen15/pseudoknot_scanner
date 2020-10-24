"""
2016-02-11, Zhewei Chen
Updated nupack_utils.py with wrapper for complexes executable

This file contains utility functions to enable easy interfacing
between Python and calls to the NUPACK core executables.  It additionally
contains utility scripts for converting structures from dot-paren
notation to pair lists.
"""
from __future__ import print_function  # for Python 2/3 compatibility
import os
import subprocess
import time
import warnings

import numpy as np
from scipy import sparse
import pandas as pd

# ############################################################### #
def energy(sequence, structure, T=37.0, material='rna1995', prefix=None, NUPACKHOME=None, delete_files=True):
    """
    Calculate the energy (in units of kcal/mol) of a particular structure for a
    specified nucleic acid sequence.

    Arguments:
    sequence -- A string of nucleic acid base codes
    structure -- A string in dot-paren format of the same length as sequence

    Keyword Arguments:
    T -- Temperature in degrees Celsius (default 37.0)
    material -- The material parameters file to use (default 'rna1995')
    prefix -- The file prefix to be used when calling the ene executable
    (default None)
    NUPACKHOME -- The file location of the NUPACK installation (default None)
    delete_files -- A flag to remove the prefix.out and prefix.in files (default
    True)

    Return:
    The energy of the nucleic acid sequence in the given secondary structure
    """

    # Check all inputs
    nupack_home = check_nupackhome(NUPACKHOME)
    material = check_material(material)
    prefix = check_prefix(prefix, 'energy')

    input_file = prefix + '.in'
    output_file = prefix + '.out'

    # Make the input file
    f = open(input_file, 'w')
    f.write('%s\n%s\n' % (sequence, structure))
    f.close()

    # Run NUPACK's energy executable
    args = [nupack_home + '/bin/energy',
            '-T', str(T),
            '-material', material,
            prefix]
    with open(output_file, 'w') as outfile:
        subprocess.check_call(args, stdout=outfile)
        outfile.close()

    # Parse the output
    en = float(np.loadtxt(output_file, comments='%'))

    # Remove files if requested
    if delete_files:
        subprocess.check_call(['rm', '-f', input_file, output_file])

    return en
# ############################################################### #


# ############################################################### #
def prob(sequence, structure, T=37.0, material='rna1995', prefix=None, NUPACKHOME=None, delete_files=True):
    """
    Calculate the probability of a nucleic acid sequence adopting a particular
    secondary structure.

    Arguments:
    sequence -- A string of nucleic acid base codes
    structure -- A string in dot-paren format of the same length as sequence

    Keyword Arguments:
    T -- Temperature in degrees Celsius (default 37.0)
    material -- The material parameters file to use (default 'rna1995')
    prefix -- The file prefix to be used when calling the ene executable
    (default None)
    NUPACKHOME -- The file location of the NUPACK installation (default None)
    delete_files -- A flag to remove the prefix.out and prefix.in files (default
    True)

    Return:
    The probability of the nucleic acid sequence adopting the given secondary
    structure
    """

    # Check all inputs
    nupack_home = check_nupackhome(NUPACKHOME)
    material = check_material(material)
    prefix = check_prefix(prefix, 'prob')

    input_file = prefix + '.in'
    output_file = prefix + '.out'

    # Make the input file
    f = open(input_file, 'w')
    f.write('%s\n%s\n' % (sequence, structure))
    f.close()

    # Run NUPACK's prob executable
    args = [nupack_home + '/bin/prob',
            '-T', str(T),
            '-material', material,
            prefix]
    with open(output_file, 'w') as outfile:
        subprocess.check_call(args, stdout=outfile)
        outfile.close()

    # Parse the output
    pr = float(np.loadtxt(output_file, comments='%'))

    # Remove files if requested
    if delete_files:
        subprocess.check_call(['rm', '-f', input_file, output_file])

    return pr
# ############################################################### #


# ############################################################### #
def mfe(sequence, T=37.0, material='rna1995', prefix=None, NUPACKHOME=None, delete_files=True):
    """
    Calculate the MFE and MFE structure of a nucleic acid sequence.

    Argument:
    sequence -- A string of nucleic acid base codes

    Keyword Arguments:
    T -- Temperature in degrees Celsius (default 37.0)
    material -- The material parameters file to use (default 'rna1995')
    prefix -- The file prefix to be used when calling the ene executable
    (default None)
    NUPACKHOME -- The file location of the NUPACK installation (default None)
    delete_files -- A flag to remove the prefix.mfe and prefix.in files
    (default True)

    Return:
    The MFE and MFE structure in dot-paren format. If there are multiple MFE
    structures, only one is returned.
    """

    # Check all inputs
    nupack_home = check_nupackhome(NUPACKHOME)
    material = check_material(material)
    prefix = check_prefix(prefix, 'mfe')

    input_file = prefix + '.in'
    output_file = prefix + '.mfe'

    # Make the input file
    f = open(input_file, 'w')
    f.write(sequence + '\n')
    f.close()

    # Run NUPACK's mfe executable
    args = [nupack_home + '/bin/mfe',
            '-T', str(T),
            '-material', material,
            prefix]
    subprocess.check_call(args)

    # Parse the output
    df = pd.read_csv(output_file, header=None, comment='%', names=['col'],
                     error_bad_lines=False)
    en = float(df.col[1])
    struct = df.col[2]

    # Remove files if requested
    if delete_files:
        subprocess.check_call(['rm', '-f', input_file, output_file])

    return struct, en
# ############################################################### #


# ############################################################### #
def pairs(sequence, T=37.0, material='rna1995', prefix=None, NUPACKHOME=None, delete_files=True):
    """
    Calculate the probabilities of all possible base pairs of a nucleic acid
    sequence over the ensemble of unpseudoknotted structures.

    Argument:
    sequence -- A string of nucleic acid base codes

    Keyword Arguments:
    T -- Temperature in degrees Celsius (default 37.0)
    material -- The material parameters file to use (default 'rna1995')
    prefix -- The file prefix to be used when calling the ene executable
    (default None)
    NUPACKHOME -- The file location of the NUPACK installation (default None)
    delete_files -- A flag to remove the prefix.ppairs and prefix.in files
    (default True)

    Return:
    The pair probabilities matrix right-augmented with the unpaired
    probabilities.

    Notes:
    Pseudoknots are not allowed.
    Pair probability matrices are symmetric, by definition, except for the
    unpaired probability column).
    """

    # Check all inputs
    nupack_home = check_nupackhome(NUPACKHOME)
    material = check_material(material)
    prefix = check_prefix(prefix, 'mfe')

    input_file = prefix + '.in'
    output_file = prefix + '.ppairs'

    # Make the input file
    f = open(input_file, 'w')
    f.write(sequence + '\n')
    f.close()

    # Run NUPACK's mfe executable
    args = [nupack_home + '/bin/pairs',
            '-T', str(T),
            '-material', material,
            prefix]
    subprocess.check_call(args)

    # Parse the output
    df = pd.read_csv(output_file, header=None, comment='%',
                     names=['i', 'j', 'p'], delimiter='\t')

    # Build a sparse pair probabilities matrix with indexing starting at 0
    # df.ix[0] is garbage.  df.i[1] is # of bases.  Indices 2 and up are probs.
    P = sparse.csc_matrix((df.p[1:], (df.i[1:] - 1, df.j[1:] - 1)),
                          shape=(int(df.i[0]), int(df.i[0])+1))

    # Fill in lower triangle (ignore sparse efficiency warning)
    # Note: don't have to worry about diagonal; it's necessarily zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P[:, :-1] = P[:, :-1] + P[:, :-1].transpose()

    # Remove files if requested
    if delete_files:
        subprocess.check_call(['rm', '-f', input_file, output_file])

    return np.array(P.todense())
# ############################################################### #


# ############################################################### #
def dotparens_2_pairlist(structure):
    """
    Convert a dot-paren structure into a list of pairs called pairlist
    and an array, plist, whose entry at index i is the base paired to base i.

    Argument:
    structure -- A string in dot-paren format

    Return:
    pairlist -- An array of ordered pairs for each base pair
    plist -- An array such that plist[i] is paired with i

    Notes:
    Array indexing in Python is zero-based.
    plist[i] is -1 if base i is unpaired.
    This only works for single-stranded structures, and not complexes with
    multiple strands.

    Example:
    structure = '.(((...)))'

    pairlist = np.array([[1,9],[2,8],[3,7]])
    plist = np.array([-1,9,8,7,-1,-1,-1,3,2,1])
    """

    # Length of the sequence
    seqlen = len(structure)

    pairlist = []
    leftlist = []
    ind = 0
    # While loop steps through list.  Each left bracket is stored.
    # Whenever we get a right bracket, it is necessarily pair with
    # the last left bracket in leftlist.  This pair is documented
    # and the first entry in leftlist is then deleted.
    while ind < seqlen:
        if structure[ind] == '(':
            leftlist.append(ind)
        elif structure[ind] == ')':
            pairlist.append([leftlist[-1], ind])
            leftlist.remove(leftlist[-1])
        ind = ind + 1

    pairlist.sort()

    # Get plist
    plist = [-1]*seqlen
    for x in pairlist:
        plist[x[0]] = x[1]
        plist[x[1]] = x[0]

    return np.array(pairlist), np.array(plist)
# ############################################################### #

# ############################################################### #
def complexes(sequence, max_complex_size, Pairs=False, Ordered=False, custom_complex=None, T=37.0, material='rna1995', prefix=None, NUPACKHOME=None, delete_files=True):
    """
    Calculate the partition functions of all strand complexes up to a specified size

    Argument:
    sequence -- array of strings containing sequences of nucleic acid base codes
    max_complex_size -- max size each strand will bind to each other. Dimer = 2, Trimer = 3, etc...
    custom_complex -- array of strings specifying additional custom complexes to calculate (default None)

    Keyword Arguments:
    T -- Temperature in degrees Celsius (default 37.0)
    material -- The material parameters file to use (default 'rna1995')
    prefix -- The file prefix to be used when calling the ene executable (default None)
    NUPACKHOME -- The file location of the NUPACK installation (default None)
    delete_files -- A flag to remove the prefix.ppairs and prefix.in files (default True)

    Return:
    out_cx -- contains a 2D array of complexes.

    Notes:

    Example:

    """
    # Check all inputs
    nupack_home = check_nupackhome(NUPACKHOME)
    material = check_material(material)
    prefix = check_prefix(prefix, 'complexes')

    input_file = prefix + '.in'
    input_file2 = prefix + '.list'
    output_file = prefix + '.cx'
    output_file2 = prefix + '.cx-epairs'
    output_file3 = prefix + '.ocx'
    output_file4 = prefix + '.ocx-epairs'
    output_file5 = prefix + '.ocx-key'
    output_file6 = prefix + '.ocx-ppairs'

    # Write input sequences
    f = open(input_file, 'w')
    f.write(str(len(sequence))+'\n')
    for i in range(0,len(sequence)):
        f.write(sequence[i]+'\n')
    f.write(str(max_complex_size)+'\n')
    f.close()

    # Write list file for custom complexes
    f = open(input_file2, 'w')
    if custom_complex!=None:
        for i in range(0,len(custom_complex)):
            f.write(custom_complex[i]+'\n')
    f.close()

    # Run NUPACK's complexes executable
    if Pairs and Ordered:
        args = [nupack_home + '/bin/complexes','-T', str(T), '-material', material,'-dangles','none','-ordered','-pairs', prefix]
    elif Pairs:
        args = [nupack_home + '/bin/complexes','-T', str(T), '-material', material, '-pairs', prefix]
    else:
        args = [nupack_home + '/bin/complexes','-T', str(T), '-material', material,'-dangles','none', prefix]
    subprocess.check_call(args)

    # Parse the output files for data
    out_cx = BuildBlockArrayByFile(output_file,len(sequence))
    if Pairs and Ordered:
        out_key, out_pp = getCXPPairs(prefix,sequence)

    # Remove files if requested
    if delete_files:
        subprocess.check_call(['rm', '-f', input_file, input_file2, output_file,output_file2,output_file3,output_file4,output_file5,output_file6])

    if Pairs and Ordered:
        return out_cx, out_pp, out_key
    else:
        return out_cx

# For building complexes array for large files
def BuildBlockArrayByFile(filename,N):
    out=np.zeros([N,N+1])
    f=open(filename,'r')
    count=0
    for line in f:
        l = line.strip().split()
        if l[0][0] is not '%':
            # format is same for all lines, so know energy already at this point
            ene = float(l[-1])
            # convert all entries in list except energy to ints
            l = [int(x) for x in l[1:-1]]
            a,b = getIndex(l)
            out[a-1][b] = ene
            if b != 0:
                out[b-1][a] = ene
            count+=1
            if count%10000==0:
                print('Building block array',count)
    return out

# Get index of interacting blocks and return index locations
def getIndex(data):
    N=len(data)
    a=0
    b=0
    for i in range(0,N):
        if data[i]==1:
            if a<1:
                a=i+1
            elif b<1:
                b=i+1
        elif data[i]==2:
            a=i+1
            b=i+1
        # early exit if indices already found (small speedup)
        if (a > 0 and b > 0):
            break
    return a, b

# Get complex pair probabilities from ocx-ppairs file, works for complex size 2 only
def getCXPPairs(filename,seq):
    file1=filename+'.ocx-ppairs'
    file2=filename+'.ocx-key'

    # Obtain ocx key of complex size 2
    ocxkey=[]
    f=open(file2,'r')
    for line in f:
        l = line.strip().split()
        if l[0]!='%':
            if len(l)<4:
                ocxkey.append([int(l[0]),int(l[2]),0])
            else:
                ocxkey.append([int(l[0]),int(l[2]),int(l[3])])
    ocxkey=np.array(ocxkey)

    # Obtain pair probabilities data
    f=open(file1,'r')
    ocxppairs=[]
    out=[]
    cx=0
    maxN=0
    newArray=0
    count=0
    for line in f:
        l = line.strip().split()
        if l!=[]:
            if newArray==1 and len(l)==1:
                maxN=int(l[0])
                out=np.zeros([maxN,maxN+1])
                newArray=0
            elif newArray==0 and len(l)==2 and 'complex' in l[1]:
                newArray=1
                ocxppairs.append(out)
            elif l[0]!='%' and len(l)==3:
                # Obtain values
                prob = float(l[-1])
                a = int(l[0])
                b = int(l[1])
                if b>maxN:
                    b=0
                out[a-1][b]=prob
                if b!=0:
                    out[b-1][a]=prob
        # low granularity progress output
        if count%10000==0:
            print('CX PPairs parsing line',count)
        count+=1
    ocxppairs.append(out)
    ocxppairs=np.array(ocxppairs[1:])
    return ocxkey, ocxppairs

# get mfe structure array from file
def getMFEPairs(filename):
    f=open(filename,'r')
    out_mfe=[]
    out=[]
    new=0
    for line in f:
        l = line.strip().split()
        if l!=[]:
            if l[0]=='%' and len(l)>1 and 'complex' in l[1]:
                new=1
            elif l[0]!='%' and new==1 and len(l)==1:
                new=0
                N=int(l[0])
                out_mfe.append(out)
                out=np.zeros([N,N+1])
            elif l[0]!='%' and len(l)==2:
                # convert all entries in list except energy to ints
                a=int(l[0])
                b=int(l[1])
                out[a-1][b] = 1
                out[b-1][a] = 1
    out_mfe.append(out)
    out_mfe=out_mfe[1:]
    return out_mfe

# ############################################################### #

# MISC useful functions
# Count frequency of A,T,G,C in sequence
def CalcATGC(seq):
    N=len(seq)
    out=np.zeros([N,4])
    for i in range(0,N):
        A=0
        T=0
        G=0
        C=0
        cN=len(seq[i])
        for j in range(0,cN):
            if seq[i][j]=='A':
                A+=1
            elif seq[i][j]=='T':
                T+=1
            elif seq[i][j]=='G':
                G+=1
            elif seq[i][j]=='C':
                C+=1
        out[i]=[A,T,G,C]
    return out

# ############################################################### #
def check_prefix(prefix=None, calc_style=''):
    """
    Generate a prefix for a file for a NUPACK calculation.
    """

    # If prefix is provided, make sure file does not already exist
    if prefix is not None:
        if os.path.isfile(prefix + '.in') or os.path.isfile(prefix + '.out') \
           or os.path.isfile(prefix + '.mfe') \
           or os.path.isfile(prefix + '.ppairs'):
            raise ValueError('Files with specified prefix already exist.')
        else:
            return prefix
    else:
        # Generate prefix based on time.
        prefix = time.strftime('%Y-%m-%d-%H_%M_%S_')
        prefix += str(calc_style)

        # Check to make sure file name does not already exist
        prefix_base = prefix
        i = 0
        while os.path.isfile(prefix + '.in') \
                or os.path.isfile(prefix + '.out') \
                or os.path.isfile(prefix + '.mfe') \
                or os.path.isfile(prefix + '.ppairs'):
            prefix = prefix_base + '_%08d' % i
            i += 1

        return prefix

# ############################################################### #


def check_nupackhome(user_supplied_dir=None):
    """
    Validate or generate a string with the NUPACKHOME directory path.

    Notes:
    If user_supplied_dir is not None, checks to make sure the directory looks
    like NUPACKHOME and then strips trailing slash if there is one.
    """

    if user_supplied_dir is None:
        try:
            nupackhome = os.environ['NUPACKHOME']
        except KeyError:
            raise RuntimeError('NUPACKHOME environment variable not set.')
    elif user_supplied_dir.endswith('/'):
        # Strip trailing slash if there is one
        nupackhome = user_supplied_dir[:-1]

    # Make sure NUPACK looks ok and has executables
    if os.path.isdir(nupackhome + '/bin') \
       and os.path.isfile(nupackhome + '/bin/mfe') \
       and os.path.isfile(nupackhome + '/bin/pairs') \
       and os.path.isfile(nupackhome + '/bin/energy') \
       and os.path.isfile(nupackhome + '/bin/prob'):
        return nupackhome
    else:
        raise RuntimeError('NUPACK not compiled in %s.' % nupackhome)


def check_material(material):
    """
    Check material input.

    Notes:
    The 'rna1999' parameters will not work with temperatures other than 37.0
    degrees C.
    """

    if material not in ['rna1999', 'dna1998', 'rna1995']:
        print('!! Improper material parameter. Allowed values are:')
        print('!!   ''rna1999'', ''rna1995'', ''dna1998''.')
        raise ValueError('Improper material parameter: ' + str(material))
    else:
        return material
