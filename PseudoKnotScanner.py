# Zhewei Chen
# 2015-12-01
# Pseudoknot scanner script
# For collaboration with Mitch Guttman lab

#Import libraries
from __future__ import print_function
from nupack_utils import *
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.signal
import scipy.fftpack
import skimage as ski
import skimage.morphology

# Thermodynamic parameters
material = 'rna1995' # Specify that we are designing RNA with rna1995 parameter set
T = 37 # Temperature we're designing with in deg. C
allowGU = 1 # Allow wobble pairs.

# Sequences used for analysis
LBRS1='GGGCTGTGGATACCTGCTTTAAATTTTTTTTTTCACGGCCCAACGGGGCGCTTGGTGGATGGAAATATGGTTTTGTGAGTTATTGCACTACCTGGAATATCTATGCCTCTTATTTGCGTGTACTGTTGCTGCTGATCGTTTGGTGCTGTGTGAGTGAACCTATGGCTTAGAAAAACGACTTTGCTCTTAAACTGAGTGGGTGTTCAGGGCGTGGAGAGCCCGCGTCCGCCATTATGGCTTCTGCGTGATACGGCTATTCTCGAGCCAGTTACGCCAAGAATTAGGACACCGAGGAGCACAGCGGACTGGATAAAAGCAACCAATTGCGCTGCGCTAGCTAAAGGCTTTCTTTATATGTGCGGGGTTGCGGGATTCGCCTTGATTTGTGGTAGCATTTGCGGGGTTGTGCTAGCCGGAAGTAGAAAGCCAAGGAGTGCTCGTATTAGTGTGCGGTGTTGCGCGGAAGCCGCAGAGGACTAGGGGATAGGGCTCAGCGTGGGTGTGGGGATTGGGCAGGGTGTGTGTGCATATGGACCCCTGGCGCGGTCCCCCGTGGCTTTAAGGGCTGCTCAGAAGTCTATAAAATGGCGGCTCGGGGGCTC'

LBRS2='CCTCAAGAAGAAGGATTGCCTGGATTTAGAGGAGTGAAGAGTGCTGGAGAGAGCCCAAAGGGACAAACAATCCCTATGTGAGACTCAAGGACTGCCAGCAGCCTATACAGCTACATTACATCTCAGCAGAACTTCTCTTCAAGTCCTCGCTACTCTGAACAAAAAGCTTACAGGCCACATGGAGAAAAAAAGATCTCCCCCCAGAATTGTGGGCTTGCTGCTTTGCAGTGCTGGCGACCTATTCCCTTTGACGATCCCTAGGTGGAGATGGGGCATGAGGATCCTCCAGGGGAATAGCTCACCACCACTGGGCAACAGGCCTAGCCCAGATTTCAGTGAGACGCTTTCCTGAACCCAGCAAGGAAGACAAAGGCTCAAAGAATGCCACCCTACATCAAAGTAGGAGAAAAGCTGCTGCAATAGTGGCACTGACCTTCGAGGAAGCCATTCTGCTCTATTTGGTTCTCTCTCCAGAAGCTAGGAAAGCTTTGCCAGCTGTTTACATACTTCAAGATGCACTGCTACCCTACTCATGCCATATAATACACAATGCCATCTACCAAATATTACCCTTCCCCAAAGCAGCACAGAAAACTGGGTCTTCAGCGTGATCAAGCAATGTGAACACACAAAAGGAAGGCAGCTTTATAAATGACCCGAGGATCAA'

LBRS3='CTGCCCTGAGAAATAATCTAAGACAAAATACATCATTCCGTCCGGTCAGGATTCAAGTGGCTCTGAAGTGAACGCCCAAGTAGAAGACAGAAGTTTTGCGACTTGAGATTTAAAAGGACCAAAATACACAGATGGCCCGTCTTGAGCTGGCTGGACAGAATGCTGACAACCCAAAGAAGAGGAACTGTTTCTACAGGACACCTGTGACTTCCAAGAGCGGGGAACTACGTATGTCATAAGACACAAAACCTGAGCTAAGTCCA'

Full='CGGCTTGCTCCAGCCATGTTTGCTCGTTTCCCGTGGATGTGCGGTTCTTCCGTGGTTTCTCTCCATCTAAGGAGCTTTGGGGGAACATTTTTAGTTCCCCTACCACCAAGCCTTATGGCTTATTTAAGAAAACATATCAAAATTCCACGAGATTTTTGACGTTTTGATATGTTCTGGTAAGATTTTTTTTTTGACATGTCCTCCATACTTTTTGATATTTGTAATATTTTCAGTCAATTTTTCATTTTTAAGGAATATTTCTTTGTTGTGCCTTTTGGTTGATACTTGTGTGTGTATGGTGGACTTACCTTTCTTTCATTGTTTATATATTCTTGCCCATCGGGGCCACGGATACCTGTGTGTCCTCCCCGCCATTCCATGCCCAACGGGGTTTTGGATACTTACCTGCCTTTTCATTCTTTTTTTTTCTTATTATTTTTTTTTCTAAACTTGCCCATCTGGGCTGTGGATACCTGCTTTTATTCTTTTTTTCTTCTCCTTAGCCCATCGGGGCCATGGATACCTGCTTTTTGTAAAAAAAAAAAAAAAAACAAAAAAACCTTTCTCGGTCCATCGGGACCTCGGATACCTGCGTTTAGTCTTTTTTTCCCATGCCCAACGGGGCCTCGGATACCTGCTGTTATTATTTTTTTTTCTTTTTCTTTTGCCCATCGGGGCTGTGGATACCTGCTTTAAATTTTTTTTTTCACGGCCCAACGGGGCGCTTGGTGGATGGAAATATGGTTTTGTGAGTTATTGCACTACCTGGAATATCTATGCCTCTTATTTGCGTGTACTGTTGCTGCTGATCGTTTGGTGCTGTGTGAGTGAACCTATGGCTTAGAAAAACGACTTTGCTCTTAAACTGAGTGGGTGTTCAGGGCGTGGAGAGCCCGCGTCCGCCATTATGGCTTCTGCGTGATACGGCTATTCTCGAGCCAGTTACGCCAAGAATTAGGACACCGAGGAGCACAGCGGACTGGATAAAAGCAACCAATTGCGCTGCGCTAGCTAAAGGCTTTCTTTATATGTGCGGGGTTGCGGGATTCGCCTTGATTTGTGGTAGCATTTGCGGGGTTGTGCTAGCCGGAAGTAGAAAGCCAAGGAGTGCTCGTATTAGTGTGCGGTGTTGCGCGGAAGCCGCAGAGGACTAGGGGATAGGGCTCAGCGTGGGTGTGGGGATTGGGCAGGGTGTGTGTGCATATGGACCCCTGGCGCGGTCCCCCGTGGCTTTAAGGGCTGCTCAGAAGTCTATAAAATGGCGGCTCGGGGGCTCCACCCGAGGCTCGACAGCCCAATCTTTGTTCTGGTGTGTAGCAATGGATTATAGGACATTTAGGTCGTACAGGAAAAGATGGCGGCTCAAGTTCTTGGTGCGGTATAACGCAAAGGGCTTTGTGTGTCACATGTCAGCTTCATGTCTGAGTTAGCCTGGAGAGGTGGCACATGCTCTTGAATGTGTCTAAGATGGCGGAAGTCATGTGACCTGCCCTCTAGTGGTTTCTTTCAGTGATTTTTTTTTTGGCGGGCTTTAGCTACTTGGCGGGCTTTGCCCGAGGGTACACTTGGTGCATTATGGTAGGGTGTGGTTGGTCCTACCTTGTGCCACTCGAAGCTGAGGCAAGGCTAAGTGGAAGTGTTGGTTGCCACTTGACGTAACTCGTCAGAAATGGGCACAAGTGTGAAAGTGTTGGTGTTTGCTTGACTTCCAGTTAGAAATGTGCATTATTGCTTGGTGGCCAGGATGGAATTAGACTGTGATGAGTCACTGTCCCATAAGGACGTGAGTTTCGCTTGGTACTTCACGTGTGTCTTTAGTCATCATTTTTTCGAAGTGCCTGCCCAGGTCGGGAGAGCGCATGCTTGCAATTCTAACACTGAAGTGTTGGATGATGTCGGATCCGATTCGAGAGACCGAGGCTGCGGGTTCTTGGTCGATGTAAATCATTGAAACCTCACCTATTAAAAGAAAGAAAAGTATCTAAGGCCATTTCAAGGACATTTGACTCATCCGCTTGCGTTCATAGTCTCTTACAGTGCTCTATACGTGGCGGTGCAAACTAAAACTCAGCCCGTTCCATTCCTTTGTATTGTTCAGTGGCTAGTCTACTTACACCTTGGCCTCTGATTTAGCCAGCACTGATCTCAAGCGGTTCTCTAAGCCTACTGGGTATAAGTGGTGACTTTGGCCAGAGTCATAGTGGATCACAAATCACTGGTGAAGAGGTAGAATCCTACCTTCTTCCAAAATCTACCCCATGACTATTGCTGGGGTTGCATTTTGATTTCAATGAATATTTTGGATGCCAACGACACGTCTGATAGTGTGCTTTGCTAGTGTTTGAATTTAAAACCGAAGTGATTGTTTTCAAAATGTATTTACGATTTGCTTACTTGTTGAATTCATTTTAATTACCTTTAGTGAATTGTTACTTTGGAGTCCTTAAAGTTTTCAATAATTTTTTTGGCAGATGATACTCAAATTACTTGGCACTTAAATGTACTTTCTTTCAAACTCATCCACCGAGCTACTCTTCAAATTTTTAAGTCTTATAACACAGATACTGTTAATGTAAAGTGAACATTATGACTGGATGTCAGGAGTATTTGAGGTTCTATACCAGTTCAGGCTTTGCTTTTGTTGCTATTGTTGATGCTATATTGACTAATGGTTTTACTTGTCAGCAAGAGCCTTGAATTGTAATGCTCTGTGTCCTCTATCAGACTTACTGTTATAATAGTAATATTAAGGCCTACATTTCAACTTTCTGTGTGTTCTTGCCTTTATGGCATCTAGATTCTCCTCAAGACTCAGCAAATAGTGCTGCTGCTATTGCTGCCCCAGCCCCAGGCCCAGCCCCAGCCCCTGCCCCAGCCCCAGCCCCAGCCCCTGCCCCAGCCCCAGCCCCTGCCCCTGCCCCAGCCCCTGCCCCAGCCCCAGCCCCAGCCCCTACCCCTGCCCCTGCCCCTGCCCCACCCAACCAACCCAATCCAGTCCAGCCCTGCCCCAGCCCAGTCCTAGCCCCAGGCCCAGATACTTTCAGACCTATCCCAAGCCCACTTCTACTTAGAGAAATTCGAATCTTCATTGATTCAGTGCTAAAATGCAGTGTCCATCACTCAGCCTATAAGACTGAGACAGCCCATCTATACCCCCTCCATACTGACTTCTAGAGTCATGGAATTTCACTTAATGCATAGAATCGTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGATAGCCCATCTATACCCCCTCCATACTGACTTACAGAGTCATGGAGTTTCACTTAATGCATGCAGTCCTATTGCTAAAATGCAGTGCCCATAACTCAGCCTATAAGACTGAGATAGCCCATTTATACCCCATACCCCCTCCATACTGACTTCTAGGGTCATGGAATTTCACTTAATACATAGAATCGTATTGCTAAAATGCAGTGTCCATCACTCAGTCTATAAGACTGAGATATCCCTATGTATACCCCATACTCCCTCCATACTGACTTCCAGAGTCATAGAATTTCACTTTGCATACGGTCCTATTGCTAAAATGCAGTGTCCATCACTCAGTCTATAAGACTGAGATATCCCTATGTATACCCCATACTCCCTCCATACTGACTTCCAGAGTCATAGAATTTCACTTTGCATACGGTCCTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGATAGCCCATCTATACCCCCTCCATACTGACTTCCAGAGTCATGGAATTTCACTTAATGCATGCAGTCCTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGATAGCCCATCTATACCCCATACCCCCTCCATACTGACTTCCAGAGTCATGGAATTTCACTTAATGCATGCAGTCCTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGATAGCCCATCTATACCCACTCCATACTGACTTCCAGAGTCATGGAATTTCACTTAATGCATGCAGTCCTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGATAGCCCATCTATACCCACTCCATACTGACTTCCAGAGTCATGGAGTTTCACTTAATGCATGCAGTCCTATTGCTAAAATGCAGTGCCCATAACTCAGCCTATAAGACTGAGATAGCCCATTTATACCCCATACCCCCTCCATACTGACTTCTAGGGTCATGGAATTTCACTTAATGCATAGAATCGTATTGCTAAAATGCAGTGTCCATTACTCAGCCTATAAGACTGAGATATCCCTATGTATACCCCATACCCCCTCCATACTGACTTCCAGAGACATAGAATTTCACTTTGCATACGGTCCTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGATATCCCTATCTATACCCTCTACCCCCTCCATACTGACTTCCAGAGTCATGGAATTTCACATAATGTATAGATTTCTATTGCTAAAATGCAGTGCCCATAACTCAGCCTATAAGACTGAGATAGCCCATCTATACCCCCTCCATACTGAGTTCCAGAGTCATGGAATTTCACTTAATGCATAGAATCGTATTGCTAAAATGCAGTGCCCATCACTCAGCCTATAAGACTGAGCCCATCTATACCCCATACCCCCTCCATACTGACTTCCAGAGTCATGGAATTTCACTTTGCATACAGTCCTACTTTACTTGTCCATGGACAAGTAAACAAAGAACTCTTGTCCTTCATGTTAATCAAGATACACCAATCAAACAAGAGTTTTATATCAGAGACTTGCCATGGAGGTATCATCTCTCAAGTCTCCTTTCCTTTAAGGAAAGAAAACCATTCTGTCATTGCTGTAGTAGTCACAGTCCCAAGTTTCTAAGCAGTGTTCAGTCGTCTTTTCTCATGTATTACCTTGAGTACTGAATAATTCTGTCAGAAATATTTTGTCCATTGGATTAGACTTTAGCTAGTCCAGCCCTGTGTGCATTTAGCAAAGGGGCAAACACAGGTCTGTTATCAGACAGTTAAAGTGCTCAGTCCCAATTTTCAAGGCATTGGCCATTAAAGGGGGTAGAATACTATATACTGTTGGCATGCTGTCATGGGTGCTATCGCCCCAGGTCACATCTTTCTAACTGATGGAGATACATTTATTTGCTCATGATATTGTATACTAGTCTCACATGCTTTCTTATTTCAGCCAAAAACCTCTGCACTGGAACATTTTATGTGGATAATCCTGACTAGGAATTGAGTCTTTTCTCAAGGTCCTAATACTACCCTTGCTTTATGTAAAGAGGGTGCTGATTACTTAATGCCTCTTACACAATTGTGCAAAATTGCAGTTGTTCAAGTCCCCTTCTGTTAGTAACCAAGATCCCATACCCTCATACCCTAATGGGTGACAATCAAGGGTGCCAACCAATGAGACCACTTCTCTGTTCTGGTCTTTCTGCTGTGCTGGGGAATCAAACCTTGAGTCTTGTGTACGCTAGTAAAGCACTGTCATAGAGCTACAGCCCCACCGTGTGGTGGTTTGAGAGAACAGCCTCTTATGTAGCCTGGGCTGGGCGGGACTTACAGGCATTGCCACCTGTAATGTAAACATATTTGTGCCTGTTGTGTGCACAGCTGCATTTGTCCCTCTTCCTAAGCATTGGATAAAGAAACCAAACTAAGTCAAGTCATTTTGTTGGTAATCAAGAAGACCTTTGATCTGTCCTGTTTTTAACTTCCAGGCTGGCCTGGAACTTAGCATATAACCCAGGCTAGCCTTGAGCTCAGGATCTAGCCTGCGTTTAACAAGTGTTGGCATATCTGGTTCCTACCACTATGCCCTGCATGCAGTCTTTCATATTGTGAATGTGCATATGTCATTTCACTGTAGTAATCTGCATCTGGTGAAGACTTATTTGTATTGCAGCAGTATTTAAGATCCTTAACATAGTAAATGTGCACAGTGTTAACTCTATTGTACATATTCTCATGTCCACAGTTGTGCCTTTTAGATCAGGACTCCTGTACTTAGCAAAGCAAAGAGGCTCACTAATATAAAGCTTCTTTCATGAGACTATAGATTGAAACGATTCCAATACGGTCAATGGTCCTTCAAGGTAAGACTTCTGTCTCTGATCATTCATATCCTCTTTGCTTTATGGAATTATGTATGTGCTGTGCACTTGAAACCCCTTCCTCAAACTATTTATGTACATACTGGCAATTTTAGTAGGATCAATTTTACTCTTAACTTTGAAGTACAGAAGTGGTGTTGACCTATAAGGTCCCATTTTGTGGCTTGCTAATAATAATGACTGATTGTAGTAGGCCTTTTCTGTTCACTACAGAAGGAAACCTGAACAGCGTAAAACTGTAATGGCCATAAACATGTACCTTGCATATTAGTATGCATTTACTGCACACATCTCATTCCATTTGGATACGATCCTACTCTCAAACCCTTTTGCAGTACAGCAAGGGTCACTAATCTTTTGGCTTCTTCATCTTCCTGGACACTGGATAAGGCTGTCCCCTCCTTTCCACTCTTTAATTTCCAGGACTATTACTTTAAAGACTTAATATTTGCATAAAGGATGGGGTTTTTAATTGATAACATGTCCCTTGAACATTAATGTATATAACAGGGACATGATCCATTCATTTTAATAAAAATACTTGGCCAGTTAATGTGTAAAATTACACTTATCCACAACCTTATTACTTTTCGGACCATTGTATCTCTTGCACTCCTGCAAGGGATACCGTTTATCTCCCAAGGTCCCTGCTAGTGGACCATTAATATACAGTGAATCTTCCTTTGTCTTTGCCAGTAAACAAAGGCCATACTCCTTCGCCTTTCATTTGCACTATATCAGGATATGCTGATCAACAAGGCCGCATTCTTTTGGACTGTTATCATATATTAAATGTATGCGTATGCACTGCCACCTGCTCTGTGCACTTGAAAGGATCCCACTCACTTCCTTAGCACCTTCAGCAGGAAGTGATAATAAGCTCAAGACTTTCATTTGGAAAGTTCACATGTCTAAGCACTTCTCTAAGAACTACTGTACCCTCTTCTCCGCTTTAAAGCAGAAAGAGGGTTGTACGAAGTGCTCTTCATTTGGACTTAAGTGCATTAATGCAGTTAGTTGTCCATCATTACCTTTGGAGTTGGATTTTACATCCTTGTACTCTTTTGACACCAGAGGCATATTAATTATTTCTGAGCACTTCTCTTGTCAATATTAATCTGTACCCTTACACATATGACCTGTGCGGCAGCAAAGGTTCTGAAATGCCTACCTTTTGACTGGGGCTGCTGAGTGGTAGTAACTATTAGTAACCTCAGCATTTGGATGATTACTATGCAAAAATGTCAAGGACCTGTGTGCTCTCTTTGCATACCATCAAGGCTACTGAGTCCCAGAATTAATTGCTAAGTTATGCGTATTTATAACTATGAATGTCTGGAATATTTTGTCCCCTTTACATTATTGCAGAGGTTGCTGAGCCCCCGAAACTACCCGGTACTGTCAATGAGCACAGGGGCTCTGACGAATGACCTGCTCTCTTCCTTAAACTGATTTTGGGACTCTTAATAGGCACAATGGCAGTTCTGGATGGTTTATTTTCTACTCCAACTTGAGCAAATCCCCTGCTAGTTTCCCAATGATATAATAAAGTACAGCAGTATGTACACCCAACAATGACCCGGATTTCGACCCTTTTTGCATTGCTTTAATATATACAATCCTAAATAGTCACAATCTCACACTTTATAGTGTTCCTTTTGCCCGGCCTCTAGTTTGTCCATTGACCACTTTTCTGAATCACTAATTCTCACAAACCCATCATTAAGGAAGAGTTTGTGCCCTTTCTCAATTCCATCATGCCATCCCTTTTGCCTCTTTGTTTGAACAGTATTGACTGGGCAAAGCCCTTCTCTTGACTTAAAGTCAACAACACCAGTTTACTCACTTCATATGGCTACAGTGTCTCAGTTGCCTTCTCCTTGCTCCCACTGAACAGAGACACCTCGAATTCTTACATTATTCTGGGTAATGTTAATTACCCCAAACACCCTATGTGTCATTAATAAATTTTGGTGTATTTATACACTGAATAGCAAAAGCAGGCCAAAACTAGGTGGATGAGCCTTCAATCTTTAACTTGCACTTCTAAATTATTCCAATTCCAACTGCTGGCACATTCTAGGGCCAGGAACCATTCTTGCCTACCTTTATTAATGCTTTATTGTGCAAAATATTGCAGGCAAGTAGCTCAGGGAGTTGGATTGCCACCTTTTACTTGGGGCTTTCCTTTACAGTATGAACTGAAAATTGTCTTCCTGAGAAGGAAGCTTAGCACTTTTCTTTCCGTTCTTCCTCCAGGAAGGAGCCAACTGTCTGCTTAAGAAACTTTAAGCCCGATTTTGTATATTGCTACTGTACAGGACCAACTGCCAGAAAAGTTATTGATAATTTTATTCCTTAAGAAAGGCATTTGGATTGCAAGGTGGATTGACTGTGAGATCATTAGCTTTTGTGAAGTAAAAATAGCCATTTGTGTCATGTTTCTGAAGACTAAGCAGTGTCTCAGTGTACTGAGGGTGATGAGTCTGTGGAAAGATCAGTGCAACTATTGCAGAATGTTTAAGACAAGTATCTTTGCTTGGTCTTTACTACAAGTTTAACAAAACGAAAAAGTCAATCTTTGTGTGGCCTTTAGTATGATTAACTTTTTGGAAGATGACCTAAGCCTTCTAATCATTATATTTTGTCTGACATTGGTCACCAGTCCTTGCTTATTTTTAAAAGGTGACTGGATGGATTAAATTTGAGAACATGTCAAGTCGCCTTTGAAAATTATATAGGCCATCACATTTAATTAATTCATTCTATCCACCATTAAACTCTGGCAATAATTTGAAGTAGCTTGAAAATTCCTAAAGTGGGAATTTATTTTAGAGATGATAGAACCTGTTTCCCCACTTTACATTTTAAAATATGTCTGCCAGGATCTAATCATTCCTTTAAACGTACACTTCAAAGAGAGATTTTCCTAGTAAGAAAAGAGCTTTCTCTAGTGTGAAGGGTGCTTTGTAGCCGCCGAGTACTTAGGTCTTTTTTGGGAGCTATTGTGTATGAGTGTATGTATGTGTGTGTGTACATGCATGTTGCTGCGCGCAGTCATTCATTCACATGGTGCTCAGACAACAATGGGAGCTGGTTCGTCTATCTTGTGGGTCCTGGAGATCAAAGTGAGATCATCAGGCTTGGCAGCAAGTGCCTTTACCCTCCGCGTGCCATCTTGCCATCCCGCTGCTGAGTGTTTGATATGACATTGCTGATGAAAATAATCATCACAACAGCAGTTCTCCCAGCATTACTGAGAAATGATACTATTTTTCTGAGGAGGATGTTCAAGTAACTCATCCAGTGCAGGATCCTGCTTGAACTACTGCTCCTCCGTTACATCAGACTCTGGCTGTTTAGACTACAGGATGAATTTGGAGTCTGTTTTGTGCTCCTGCCTCAAGAAGAAGGATTGCCTGGATTTAGAGGAGTGAAGAGTGCTGGAGAGAGCCCAAAGGGACAAACAATCCCTATGTGAGACTCAAGGACTGCCAGCAGCCTATACAGCTACATTACATCTCAGCAGAACTTCTCTTCAAGTCCTCGCTACTCTGAACAAAAAGCTTACAGGCCACATGGAGAAAAAAAGATCTCCCCCCAGAATTGTGGGCTTGCTGCTTTGCAGTGCTGGCGACCTATTCCCTTTGACGATCCCTAGGTGGAGATGGGGCATGAGGATCCTCCAGGGGAATAGCTCACCACCACTGGGCAACAGGCCTAGCCCAGATTTCAGTGAGACGCTTTCCTGAACCCAGCAAGGAAGACAAAGGCTCAAAGAATGCCACCCTACATCAAAGTAGGAGAAAAGCTGCTGCAATAGTGGCACTGACCTTCGAGGAAGCCATTCTGCTCTATTTGGTTCTCTCTCCAGAAGCTAGGAAAGCTTTGCCAGCTGTTTACATACTTCAAGATGCACTGCTACCCTACTCATGCCATATAATACACAATGCCATCTACCAAATATTACCCTTCCCCAAAGCAGCACAGAAAACTGGGTCTTCAGCGTGATCAAGCAATGTGAACACACAAAAGGAAGGCAGCTTTATAAATGACCCGAGGATCAACATGCCTGACTGCAGCATCTTAAAAGCAATAGAATGAGTGTGTATTGTGGGTGTGTCTATTTCTTGTTTTATGTATCTATTTTTTCCTTGGTCTGTGTGTCTAATTCTTTGTTACATCTATTTCTTCCTTGCTTTGTGTGTCTATTTCTTCCTTGCTTTGTGTGTCTATTTCTTCCTTGCATTATGTCTAATTCTTTGTTATATCTATTTCTTCCTTGCTTTGTGTCTATTTCTTCCTTGCAGTTGTGTCTAATTCTTTGTTACATCTATTTCTTCCTTGCTTTGTGTGTCTATTTCTTCCTTGCATTGTGTCTAATTCTTTGTTATATCTATTTCTTCCTTGCTTTGTGTGTCTGTCTTCCTTGCTTTGTGTCTATTTCTTCCTTGCAGTTGTGTCTAATTCTTTGTTACATCTATTTCTTCCTTGCTTTTGTGTGTCTTTCTTTCTTGCTTTTGTGTGTCTATTTCTTCCTTGCAGTTGTGTCTAATTCTTTGTTACATCTATTTCTTCCTTGCTTTTGTGTGTCTATTTCTTCCTTGCATTGTGTCTAATTCTTTGGTATATATATTTCTTCATTGCTTTGTGTGTCTATGTCTCCTTGTGTTGTCTAATTCGTTGTTGCATCTATTTCTTCCTTGCTTTGTGTGTCTATTTCTTCCTTGCTTTGTGTGTCTATGTCTTCCTTGCTTTGTGTGTCTATGTCTTCCTTGTTTTGTGTATCTACTTCTTCCTTGTGTGTCTAATTCTTTGTTACATCTATTTCTTCCTTCCTTTGCATGTCTCCTTCTTTCCTTTGTGTGTCTTTTCTGTCTGCAGTGTGTCTTACCTATTCCCATGTTTCTCCTGCATGTTCTTTCTTGCAGAGCTTTGAGCTTTGTTTCACTTTCTCTGGTGCCTGTGTGGTCTGCTTTGTCTTCACTAGCTATGGCTCTCTGTTTTATCTATCTGGTTGCTATTTCTCTTAGCTTTTCTTTCACTCCTGCCTTTCGTGACTCCCCTTTGGGTCACATGTTGCATGCATCCCTCTCTTTTTCTTGTGCTCACCCCACTTGTTCTTTGTTCAAGTTCTCTTTGTCAGTCCATTTCAGTTTTCTTTCTGCTGCTTCTATCCTTAGTGAATTCTTGTTTACATTTCTTCCCTGCCTTTCTTGGGCCACTTTCTCTGTTTTCTTTTGTATTTGTGTCTCTTTGCTATTGGTGGATTTCTTATCTCAGCATCATTCTGTTGCTTTGTGTTTGCTTGTGTTTCTATCTTCTACTTTCCTCCTTTCTGTTCACTTTGAGCATTTCATCTCTTTACAAGTCTGTGTCTCTCTTGTATTCTAAAGTAATCCTTTCTTGGATGTTTCTTTGTATGTACATGTGCGTGTGTGCATGTGTGTTATGTGTGTCATGTGTGAGAGGAGCTTCATAGCCCCTTCCCAATAGGTCCAGAATGTCACCCGTGGAGCCGTTCCTCACACCAGACTGCCCTGAGAAATAATCTAAGACAAAATACATCATTCCGTCCGGTCAGGATTCAAGTGGCTCTGAAGTGAACGCCCAAGTAGAAGACAGAAGTTTTGCGACTTGAGATTTAAAAGGACCAAAATACACAGATGGCCCGTCTTGAGCTGGCTGGACAGAATGCTGACAACCCAAAGAAGAGGAACTGTTTCTACAGGACACCTGTGACTTCCAAGAGCGGGGAACTACGTATGTCATAAGACACAAAACCTGAGCTAAGTCCAAGCATAAGACCTAAGGACCCAATCCTATATGGACAGAATATTTAAGAGATAAAGGCCTATGGCCCAGAACTCTGGAAGGATATTTCTATCCTTCTATCCCCAAGACCAAGAAGGGAAATTCGAAGATGAGACCTGCCCCCCAACCCCAGCATCCCTTTCCATTTCTTATATTTCTATTTAAGCTGTCTTCACTTGAGATGTAATTTTTCATTGTTGCCATTGCCCATAAAGGAATACGTTTTTAGCTGGATAGTATTGTGCAAGGGTCTGTTTTAAACTGGGTCTTAGCCATTTGTTAAATTGTTGATGTTTTACAACTTCCATTTCTCTTCACATCTGCTCCACTTGAGACGGAACTAAATCCAGCCAGTGTATATAGCCTGACTATTGAAACTTCCCTAGGAATAAGCATGCATACAGATATGCATACTGCCATCCTCCCTACCTCAGAAGCCCTAGGCTGACAAGAAAAGGAAAGCATCAGGTTGTTAGGGGGAAAACAATGTCAGGCTATCTAGAGAAAATATAAAGAGTTGTTCCAGACCAATGAGAAGAATTAGACAAGCAATATGCAGATGTGCCAACCCTCTGAGAAGCACCAGCCAGTGTCACCTTCTTTCTTTGGGCTTAGGTGAGCAGGGTATGGTTTTCTAATAATGGTTTGGGGACAAAATGAGGTCTGAACTCCCTGCTCATAGTAGTGGCCGAGTAATTTGGTGCATTTCACCAAAGGAACTCCTGGGTCTAATACCTACCTTTAAAATTAATGATGAGAGACTCTAAGGACTACTTAACGGGCTTAATCTTTTTCGTGCCTTCCTCTTCCTCTGTAAGAGGGAAGTTAAATGACACAGGATGAAAAAGTAACATGCTCATAGCACATTGGCAATTATACATGGTTATTATCTGAAAGTGTAGAGCTTTTCCTATAAGGCATCAGACTAAGTACCTGAAGCTTTGTGGGTTCATGGTCTTAGTTGCATATTCCTTAGTTGCAAATCCTTTTCAAAAGGTAAGAAAAAGGCACACTGGTCTATTGCCTGTACTTGATCAAGCCCTGATATGAATGCCAGGGAATGTCTGAGTAACATTAATTTCCTTCCCTGCATATTTTTTGTGCTGAATACTAAGGCTGTGATGCTTCACTGTGGTCACCCCCAGGTAACAAGATATTACCAGGTAACCAGGAAACGTATGAATACGTAAACCATGAAGCCTACTGTAACTTCCAAGTCAGTGCTGAGTATGTATTACATAGTAGCTGAAGTCTACGCCTCTGTGTGCTATAGGCACAAAGATTGCTCTAGGAATAACATGCTTTGTAAAAACAAATATATGAACATAACGGGGCTTGAATGAATAACAGTCCATATACTTAAGGCCAGTGTGTTTCTTCTGCTTTGGTGAGGCTCAGTAAGTTATATTATACCAGGTAGCAGAAGAGAAAACACATGGAAACTGATTTTAAACTACAAACTAGGTCACTAATGCAGGTGATTGATTACCCTATTCTGATCACCTTCTAATTTCTGAATACCCATGTTCAGCACTGGGAATAACAAAGGGGGACATTACCACAGAACTAGAATTTACAAAAGAATGCATTAAATAAAGCATTATACAGCTATCAATTGTTCCATGTGTGCAAATGAATGACTACTAACTACCTCTGATGTATCCGATATTGTTTTGGGTACATGAAATATTCATGAGTAACTGCCATGAAATAAGAATGTTTGCATTCCATACTATTCATAAGGAATGAGCCAATGCTTAATTTAATCAGTCAAAACTTGAGTGATAAGGGCATGTTAATACAAGAACATTTGCCCAGGTCACATTATGGTTGTGGGTACTTTCTTAACTATAAAGCAGTTCAGTAGTATAAGACAAGACAAATTTTCTATAGAAATAAAGCTGCCTATAAAATAGGCATAGTCTCTACAAAATTTTCATTGTACTTTTTAGCCCATAATGGGAAGAGTACAGTTAACAAGCTGGGTGTGGTAGCATGTGCTCTGAGCTGAAGCAACAGGACCACTTGAGCCCAGAAATTGGAGGCTAGCCTGGGAAGACCATAAGGTCAATCTCAAACCTGGAGGCTAAATATTGTCTCCCATGTGTATATTCTCTTTCATGGGTACTGGAGAGATACACAGACGTACATTTCAGTGTGTCCACACTTGAGAATAATATGTACGTTGGCATTTTATGAACTCGGAGGTACCATATAAATGTAACAATTCATTTTCTTACTTGGTATCAATTTCCAGGCTTTTAAAATTCTGCCACATTTATTATACTGTGAAAATAAAGTAAATAAGTAACTGTGAACCACTGAATATATGAAGCATTCAATACTTGATGAGTACATACTGAATGGCAGTCATTTATTACAAAACAGTGCCCTTGCTAGGCACTGGGATGCAAAGAGCATTCTCATTGTCCTGTGTATCTAAAGAAATTATGCATGAGATTAATTTATAATTTGTAAACTGCCATATATATGTGTATATATGCAATATTTGCCTGGTGTGCAATGACTTTGCTTTTATCCCAGGCATGCACAACAGATCTGTGTGGAGCTTTGTGAAGTCTACAGTTCTATAAAGCCGGGACCTAACTGTTGGCTTTATCAGTGAACAGTGATTACTTTCTAAGTTTCATAATGGCTGAAACTTAATCATAATGCTTATCACCTAACACCACCTAATAATAATTTTACCATGCTATGTGTTGAGCGAACACATAGATTGCTTTCTAGCATTATGTAGCACTTATAGGAGTGAAATCTAGACCAAAACTTCAATTCACTTCAATGAGGAAATGAAAACAGAAAAAAAAAATGGATTTGTGCAAGGCAGTGTGCTAAATGTTACACTGAGTGGACTATGCTGTCTAGGATACTTCCCAGCTGGCTTGACTGAGGAGGTGGAAAAGGTTTTATTAATGACAGGAACTTTTTCCATCCAGTTTCTTAAATGTTTGTTGAATGCTGCTGCCAGAGATGAATTACAAACACCTTGCCAGTAAAGGAGTTTTATAGGGCCAGAGTGAGATAATCCCAGAGCATGGGTATCAGGGAACAAAACGGGAAGAGGCCAGAGCATCTGATGGCATGTACTCAGTGTGGCCCAGAACCTCTCGAACTAGATGTACTGGCTGGAGGGACCAAGCATGCAGAACACAACACCTAATGAAACATTGTATATAAAATATGCTAACCTAGGTCCTAAAACTAAAATGTGAGGTGGACCTAGTGTAGATCACTGATCATAGGAGACATGGTCTCATAAAGCCCAGGCTGGTTCTAATTGGTGACTGTCACAGCTTCTCAAGTGCTGAGATTACAGATGTGCTTAACCCATGCCCAGCCTGAAGAATATATCTGATTACTGAGTGAATAATATTTTTAAAGAATTATATATTTTATGTATATGAGTACGCTGTTGCTGTCTTCAGACACACCAGAAGAGGGCACCACATCACATTACAGATGGTTGTGAGCCCCCATGTGGTTGTTGGGATTTGAACTCAGGACCTTCGGAAGAGCAGTCAGACTCTTAACCACTGAGTCATCTCTCCAGCCTTCTGAGTAAATATTTTAACTATAATGGCTGTTTGCGAAACCCAACCAAGGCCAAGATTCCTTCAACATAAACTGGAGACTTCCTAGCTAAGGAAGCTCCAAAAGTCATTTTCTCATTGGCCTAGCTTGAAGCCAGGACAGACTTAAAGTCTGTCCTTTAATTCATTACCCATTTTCCTTTTCTTACTGTTGAAGTGTTTCAAAGGAGAATCAAGATGAATCGATAATTCTAAACGTATTTGTTCATTGCCTGGCTCAGCGTCATGTGAGCAAGAAGAATATACTATCACACTCATACTTTTAACTTAAGTGTGATGAAAGTGCAGTTCTAAGTACTAAAATTTCTAAGTACTGAAAAGAACAAAGACATTTAAAGGATGCAACCCAAAGTGTACTTTACCTCAGTAGTTTCTGAGGGGACTGCAGTCACACCTTGAGACTACAGCTCTCACTTTAGCTGGGAAAAACATCAAGGTGTAGAGGAGGCAAGTTAAATAAAAAGTTGCTCCCCTCCTCATGGGCATGCTTGGTAGAGTGGAAATAATAAAAGAGGTTCTCTATTTCCTCGGTTCCACACATTGCAGAAGATGCTACTGGATGCTAAGTGCAACACATTTGTTCCAAAAGGGCACTCAGTGTGACTTACAGATGCCCCGGAAAGCAGAGGGATGCTCTTTATTAAACAGAAATATTAGCTCAAACGTTTTCTAGACTGAAGAACACTTTCCTCATTTCCCACAGTTTGCCTCAGAGGTTGAATACAGGAAGGTTATTATTCATTCATTTGCTTTATTGGTTCGCCTGTTCTACAAGGATTTGCATGTCTCTTAGGCCTTCACTTGGCTCCTGAGACATGGAAAAAGGAAACATAGACATAGGGAAGTGCTGGATGGGGGGGGGGGGTCTCTTTTCTGGGTAGTGGCACGACTTAGTCCTTAGTCCCCAAGTAATATGCAATGTGAGTCCTCATCCTCATGTCTTCTCCGGCCACTGCAATGAGTGGGAAGCTGGGCTTTGTAGCAAGCCTGACCCTAAAGTTACAGAAGCCCTCCACGCTAAGAAACTCAATTTTCTAGGCCATTTTAGCTATGACTGTGACCACTACTGGTCAGGAGGGATGACAGCCATCTAAGTTCCACAATCTTAGGCTACTTTGCATTATCCTGGGGCAAACAAGCCATTTTTGAGCTGCAGCAGGCTTTGAAATACATTGACCAATTTTGCCTGTGTTCGTTAAACCTTTTACCTTTTTACATGCTAATGCTCACAGTAATTTAGAAATGTTCTCCTTACTATAATATACTCAAGGTGGCTTGCTATGGTAAAATAATGCCAGTGGATGAAAATAACATTAATGTTTAACATTCTTGCATAAAATTTAAGAATAATAAAATTGACAACAATCAGAAAACTGGAGGAACGAAAGACCAAATTGAAAGAACTTGAAAAAGATTAAAAATGCCTGTGCTTTGACCCTTTCCATTTTTCTTTCACTCACAGAGGGTGGGACAGGAGGCCGAGTGAAGGAAAGGGTCCAGCCTGTCTATCTGGAATCTAAGTTGGGACTTTAATGCAGTTCCACAAAATTGGTATTAATTCGCTAAATGTTTCTGAAAATGTATTTTCATCTAAATGGCTATCAGCTAAGCCTTGAGTCAAATGGGAATGAAACAGATTAAGTCAATGTGATCTCTTTATCCAAGTTGCCTTAGAGCTGAAGTCACAATTTGCTGTAAGGAAGCTTATTCATTGTAGCATACGCATACTTTCAAAGTATCTAGACTTTACTTAGTAACCCAATCAGGACATTCAGGCAAAAGAAAAGGAACAGAGAAGATGGAGCCAGGTTGAAGAGGTCTGGGAGTTCAAACAAATTTTTTTCATTTTCATTAAAACTCAATTGGGCATCAAAAGTGTTACTAATATTAGCTTTTAATTAGTGGAAATTGGCTGGATTCAGTAATATCCCTTTGTATGGGTAGGAATGGGCTTACATTTCTGGAATTTGCAAAGGAAAAAATAACTGAAAGCCTTCCTTTCACAGTTACTGCCATCAATATTGCTACCAATTAAGCACATCCTACCATCATCTGCTTTGATCACATAAATGAACTGTGTACCAATCTGTTGTTGAAAGACTGGAGTCATCTTCCCACCAACTGTGAAAAAACACATGGAAAACACCTGGACTTTGTGAACGGATGCGGAATACAGAACTTCTGTTGACTCTTGGGTGTTTTGAAGACTTGAAAAAAAAAACTGTTGCTTACCAACATGTCACAATGAGTCCGTGTGTGGGTGGGTGGATGGGTGGGTGGGTGGGTGGGTGGGTGGTTGAGTGGGTGGGGTAGTTTGCTGTTAAATAAAATGCTTTGTTTTGAA'

# For debugging
test='AAAAABBBBBCCCCCDDDDDEEEEEFFFFFGGGGGHHHHHIIIIIJJJJJKKKKKLLLLLMMMMMNNNNNOOOOOPPPPP'

##################################################################################################
##################################################################################################
# For building a window of blocks for pseudoknot scanning

# Generate a moving block window of size N which is centered on the index nucleotide. The index nucleotide steps down the sequence at L nucleotides
def GenerateBlockWindow(N,L,Seq):
	length=len(Seq)
	out=[]
	count=0
	index=0
	if N%2==1:
		while index<length:
			if index-N/2<0:
				block=getSeq(0,N/2+index+1,Seq)
			else:
				block=getSeq(index-N/2,N,Seq)
			out.append(block)
			index+=L
	else:
		while index<length+1:
			if index-N/2<0:
				block=getSeq(0,N/2+index,Seq)
			else:
				block=getSeq(index-N/2,N,Seq)
			out.append(block)
			index+=L
	# Clean spacers
	for i in range(0,len(out)):
		out[i]=cleanBlock(out[i])
	return out

# Clean the last block of _
def cleanBlock(Seq):
	out=[]
	for i in range(0,len(Seq)):
		if Seq[i]=='_':
			return "".join(out)
		else:
			out.append(Seq[i])
	return "".join(out)

# Grab N length sequence from index i
def getSeq(index,N,Seq):
	out=['_']*N
	maxN=len(Seq)
	for i in range(0,N):
		# If block size runs over sequence length, then return what is leftover
		if (index+i > maxN-1):
			return "".join(out)
		out[i]=Seq[index+i]
	return "".join(out)

# Break sequence in variable block sizes at indexes y
def GenerateBlockDynamic(y,Seq):
	start=0
	end=0
	out=[]
	for i in range(0,len(y)):
		size=y[i]-start
		block=getSeq(start,size,Seq)
		out.append(block)
		start=y[i]
	size=len(Seq)-start
	block=getSeq(start,size,Seq)
	out.append(block)
	return out

##################################################################################################
##################################################################################################
# For building and exporting sequence data

# Build sorted list of interacting sequences
def BuildDataList(seq,ene):
	# Calculate start index of sequence
	cxN=len(seq)
	seqstart=np.zeros(cxN)
	count=0
	for i in range(0,cxN-1):
		count+=len(seq[i])
		seqstart[i+1]=count

	# build data list on blocks and sequences
	data=[]
	for i in range(0,cxN):
		data.append([i+1,0,seqstart[i],'na',len(seq[i]),'na',ene[i,0],seq[i],'na'])

	for i in range(0,cxN):
		for j in range(i,cxN):
			data.append([i+1,j+1,seqstart[i],seqstart[j],len(seq[i]),len(seq[j]),ene[j,i+1],seq[i],seq[j]])
	return data

# Save list to file
def SaveData(outfile,data):
	f=open(outfile,'w')
	f.write('block_A\tblock_B\tindex_A\tindex_B\tlenA\tlenB\tkcal/mol\tseqA\tseqB\n')
	for i in range(0,len(data)):
		for j in range(0,len(data[i])):
			f.write(str(data[i][j])+'\t')
		f.write('\n')
	f.close()
 
##################################################################################################
##################################################################################################

# Partially rebuild pair probabilities plot from energy data and complexes pair probability data
def BuildPPplot(ppairs,key,seq,ene):
	# Calculate index offsets
	cxN=len(seq)
	offset=np.zeros(cxN)
	seqlen=np.zeros(cxN)
	for i in range(0,cxN-1):
		seqlen[i]=len(seq[i])
		offset[i+1]=offset[i]+seqlen[i]
	seqlen[cxN-1]=len(seq[cxN-1])

	N=np.sum(seqlen)
	out=np.zeros([N,N+1])
	for i in range(0,len(key)):
		a=key[i,1]-1
		b=key[i,2]-1
		lenA=seqlen[a]
		sa=offset[a]
		ea=offset[a]+lenA
 		
		dummy=np.copy(ppairs[i])*(ene[a,b+1])
		dummy2=np.copy(ppairs[i])*(ene[a,0])
		
		#dummy=-np.copy(ppairs[i])
		#dummy2=-np.copy(ppairs[i])

		# Q1
		if a==b:
			copy=np.copy(out[sa:ea,sa+1:ea+1])
			val=dummy2[0:lenA,1:1+lenA]/(lenA)*2
			out[sa:ea,sa+1:ea+1]=np.sum([val,copy],axis=0)
		elif b>-1 and b!=a:
			lenB=seqlen[b]
			sb=offset[b]
			eb=offset[b]+lenB
			# Q2
			copy=np.copy(out[sa:ea,sb+1:eb+1])
			val=dummy[0:lenA,1+lenA:]/(lenA+lenB)*2
			out[sa:ea,sb+1:eb+1]=np.sum([val,copy],axis=0)
			# Q3
			copy=np.copy(out[sb:eb,sa+1:ea+1])
			val=dummy[lenA:,1:1+lenA]/(lenA+lenB)*2
			out[sb:eb,sa+1:ea+1]=np.sum([val,copy],axis=0)
			"""
			# Q4
			copy=np.copy(out[sb:eb,sb+1:eb+1])
			val=dummy2[lenA:,1+lenA:]/(lenA+lenA)*2
			out[sb:eb,sb+1:eb+1]=np.sum([val,copy],axis=0)
			"""
	return out

# Partially rebuild pair probabilities plot from energy data and complexes pair probability data
def BuildPPplot2(ppairs,key,seq,ene):
	# Calculate index offsets
	cxN=len(seq)
	offset=np.zeros(cxN)
	seqlen=np.zeros(cxN)
	for i in range(0,cxN-1):
		seqlen[i]=len(seq[i])
		offset[i+1]=offset[i]+seqlen[i]
	seqlen[cxN-1]=len(seq[cxN-1])

	N=np.sum(seqlen)
	out=np.zeros([N,N+1])
	for i in range(0,len(key)):
		a=key[i,1]-1
		b=key[i,2]-1
		lenA=seqlen[a]
		sa=offset[a]
		ea=offset[a]+lenA
 		
		dummy=np.ones(ppairs[i].shape)*(ene[a,b+1])
		dummy2=np.ones(ppairs[i].shape)*(ene[a,0])
		# Q1
		if a==b:
			copy=np.copy(out[sa:ea,sa+1:ea+1])
			val=dummy2[0:lenA,1:1+lenA]
			out[sa:ea,sa+1:ea+1]=np.sum([val,copy],axis=0)
		elif b>-1 and b!=a:
			lenB=seqlen[b]
			sb=offset[b]
			eb=offset[b]+lenB
			# Q2
			copy=np.copy(out[sa:ea,sb+1:eb+1])
			val=dummy[0:lenA,1+lenA:]
			out[sa:ea,sb+1:eb+1]=np.sum([val,copy],axis=0)
			# Q3
			copy=np.copy(out[sb:eb,sa+1:ea+1])
			val=dummy[lenA:,1:1+lenA]
			out[sb:eb,sa+1:ea+1]=np.sum([val,copy],axis=0)
			"""
			# Q4
			copy=np.copy(out[sb:eb,sb+1:eb+1])
			val=dummy2[lenA:,1+lenA:]/(lenA+lenA)*2
			out[sb:eb,sb+1:eb+1]=np.sum([val,copy],axis=0)
			"""
	return out

##################################################################################################
##################################################################################################
# Drawing functions for plots

# Draw lines to denote LBRS1, LBRS2, LBRS3
def DrawL1L2L3():
	L1=len(LBRS1)
	L2=len(LBRS2)
	L3=len(LBRS3)
	plt.plot([L1+1-.5,L1+1-.5],[0,L1+L2+L3-.5],c='black')
	plt.plot([L1+L2+1-.5,L1+L2+1-.5],[0,L1+L2+L3-.5],c='black')
	plt.plot([0,L1+L2+L3],[L1-.5,L1-.5],c='black')
	plt.plot([0,L1+L2+L3],[L1+L2-.5,L1+L2-.5],c='black')
	plt.ylim([-0.5,L1+L2+L3-0.5])
	plt.xlim([-0.5,L1+L2+L3+0.5])

# Function to draw polymer graph based on ppairs plot
def DrawPolymerGraph(ppairs,mfe,seq):
	L1=len(LBRS1)
	L2=len(LBRS2)
	L3=len(LBRS3)
	fig=plt.gcf()
	ax=fig.add_subplot(111)

	# Specify dimensions of circle
	Spacing=5
	Circum=(L1+L2+L3)*Spacing+5
	r=Circum/(2*np.pi)
	theta=np.linspace(0,-2.0*np.pi,Circum)

	# Generates circular coordinates for each of the base pairs in polymer graph
	Coord=np.zeros([L1+L2+L3,2])
	for i in range(0,len(Coord)):
		Coord[i,0]=r*np.cos(theta[i*Spacing])
		Coord[i,1]=r*np.sin(theta[i*Spacing])
	
	# Define colormap and thresholds for drawing base pairing on polymer graph
	MaxVal=np.max(-ppairs)
	MinVal=.60

	# Draw base pairing on polymer graph
	a,b=ppairs.shape
	for i in range(0,a):
		for j in range(1,b):
			# Overlay mfe structure as black color
			if mfe[i,j]>0:
				x=[Coord[i,0],Coord[j-1,0]]
				y=[Coord[i,1],Coord[j-1,1]]
				plt.plot(x,y,c='black')
			# Plot pair probs or energy
			if -ppairs[i,j]>MinVal:
				x=[Coord[i,0],Coord[j-1,0]]
				y=[Coord[i,1],Coord[j-1,1]]
				value=-ppairs[i,j]
				#cvalue=np.abs(value-MinVal)/(MaxVal-MinVal)
				cvalue=np.abs(value)/(MaxVal)
				plt.plot(x,y,c=cm.jet(cvalue))

	# for annotating the polymer graph
	# Generate tick marks on polymer graph
	TL=150 # Tick mark length
	SP=20 # Tick mark spacing
	Tick=[]
	for i in range(0,L1+L2+L3):
		if i%SP==0:
			x1=r*np.cos(theta[i*Spacing])
			y1=r*np.sin(theta[i*Spacing])
			x2=(r+TL)*np.cos(theta[i*Spacing])
			y2=(r+TL)*np.sin(theta[i*Spacing])
			Tick.append([x1,x2,y1,y2])
	Tick=np.array(Tick)

	# Draw tick marks
	for i in range(0,len(Tick)):
		lx=[Tick[i,0],Tick[i,1]]
		ly=[Tick[i,2],Tick[i,3]]
		x=Tick[i,1]-int(np.log10(i*SP+1))*15.0 # offset the size of the text
		y=Tick[i,3]
		plt.plot(lx,ly,c='Black',linewidth=1.0)
		ax.annotate(str(i*SP),xy=(str(x),str(y)),size='small')

	# Draw boundaries for LBRS1, LBRS2, LBRS3
	plt.plot(Coord[0:L1,0],Coord[0:L1,1],c='Red',linewidth=4.0)
	plt.plot(Coord[L1+1:L1+L2,0],Coord[L1+1:L1+L2,1],c='Green',linewidth=4.0)
	plt.plot(Coord[L1+L2+1:,0],Coord[L1+L2+1:,1],c='Cyan',linewidth=4.0)

	# Annotate block boundaries
	# Calculate start index of sequence
	cxN=len(seq)
	blockloc=np.zeros(cxN)
	count=0
	for i in range(0,cxN-1):
		count+=len(seq[i])
		blockloc[i+1]=count

	# Generates new circular coordinates for block boundaries
	BL=50 # how much to offset outer circle
	TL2=20 # new tick mark
	Coord=np.zeros([L1+L2+L3,2])
	for i in range(0,len(Coord)):
		Coord[i,0]=(BL+r)*np.cos(theta[i*Spacing])
		Coord[i,1]=(BL+r)*np.sin(theta[i*Spacing])
	plt.plot(Coord[:,0],Coord[:,1],c='Black',linewidth=2.0)

	# add tick marks to outer circle to demark block boundaries
	Tick=[]
	for j in range(0,len(blockloc)):
		i=blockloc[j]
		x1=(r+BL-TL2)*np.cos(theta[i*Spacing])
		y1=(r+BL-TL2)*np.sin(theta[i*Spacing])
		x2=(r+BL)*np.cos(theta[i*Spacing])
		y2=(r+BL)*np.sin(theta[i*Spacing])
		Tick.append([x1,x2,y1,y2])

	i=0
	x1=(r+BL-TL2)*np.cos(theta[i*Spacing])
	y1=(r+BL-TL2)*np.sin(theta[i*Spacing])
	x2=(r+BL)*np.cos(theta[i*Spacing])
	y2=(r+BL)*np.sin(theta[i*Spacing])
	Tick.append([x1,x2,y1,y2])

	i=L1
	x1=(r+BL-TL2)*np.cos(theta[i*Spacing])
	y1=(r+BL-TL2)*np.sin(theta[i*Spacing])
	x2=(r+BL)*np.cos(theta[i*Spacing])
	y2=(r+BL)*np.sin(theta[i*Spacing])
	Tick.append([x1,x2,y1,y2])

	i=L2
	x1=(r+BL-TL2)*np.cos(theta[i*Spacing])
	y1=(r+BL-TL2)*np.sin(theta[i*Spacing])
	x2=(r+BL)*np.cos(theta[i*Spacing])
	y2=(r+BL)*np.sin(theta[i*Spacing])
	Tick.append([x1,x2,y1,y2])

	Tick=np.array(Tick)

	# Draw tick marks
	for i in range(0,len(Tick)):
		lx=[Tick[i,0],Tick[i,1]]
		ly=[Tick[i,2],Tick[i,3]]
		plt.plot(lx,ly,c='Blue',linewidth=2.0)

##################################################################################################
##################################################################################################

# Function to search for interacting sequences via complexes function
def PseudoKnotSearch():
	N=19
	Step=1
	seq1=GenerateBlockWindow(N,Step,LBRS1)
	seq2=GenerateBlockWindow(N,Step,LBRS2)
	seq3=GenerateBlockWindow(N,Step,LBRS3)
	seq=np.concatenate((seq1,seq2,seq3))
	print('Sequence generation complete')
	L1=len(seq1)
	L2=len(seq2)
	L3=len(seq3)

	"""
	# Calculate free energies
	msize=2
	out=complexes(seq, msize)
	np.save('out_EnergyWindow.npy',out)
	"""

	#Build data array
	print('Building data array')
	out=np.load('out_EnergyWindow.npy')

	plt.figure(1)
	plt.imshow(-out,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Pseudoknot search plot')
	plt.savefig('Figure1.png',dpi=300)

	# GC content vs stickyness of block
	plt.figure(2)
	plt.title('Block stickiness vs nucleotide composition')
	ATGC=CalcATGC(seq)
	sticky=sum(out<-10)/100.0

	plt.subplot(221)
	plt.xlabel('#Delta_G threshold x100')
	plt.ylabel('#of A in block')
	plt.hist2d(sticky[1:],ATGC[:,0],bins=50)
	plt.colorbar()

	plt.subplot(222)
	plt.xlabel('#Delta_G threshold x100')
	plt.ylabel('#of T in block')
	plt.hist2d(sticky[1:],ATGC[:,1],bins=50)
	plt.colorbar()

	plt.subplot(223)
	plt.xlabel('#Delta_G threshold x100')
	plt.ylabel('#of G in block')
	plt.hist2d(sticky[1:],ATGC[:,2],bins=50)
	plt.colorbar()

	plt.subplot(224)
	plt.xlabel('#Delta_G threshold x100')
	plt.ylabel('#of C in block')
	plt.hist2d(sticky[1:],ATGC[:,3],bins=50)
	plt.colorbar()

	plt.tight_layout()
	plt.savefig('Figure2.png',dpi=200)

def OverlayPairsPlot():
	L1=len(LBRS1)
	L2=len(LBRS2)
	L3=len(LBRS3)
	print(L1,L2,L3)
	N=L1+L2+L3

	# Parse ppairs data from NUPACK web interface
	"""
	out=np.zeros([N,N+1])
	clist=np.loadtxt('L1L2L3_data.txt',comments='%', delimiter='\t')
	cN=len(clist)
	for i in range(0,cN):
		a=clist[i][1]
		if clist[i][0]==2:
			a+=L1
		elif clist[i][0]==3:
			a+=L1+L2

		b=clist[i][3]
		if clist[i][2]==2:
			b+=L1
		elif clist[i][2]==3:
			b+=L1+L2
		elif clist[i][2]==-1:
			b=0
		out[a-1,b]=clist[i][4]
		# make matrix symmetric
		if b!=0:
			out[b-1,a]=clist[i][4]

	#Save data after calculation
	np.save('out_nupack.npy',out)
	"""

	pout=np.load('out_tpp.npy')
	eout=np.load('out_EnergyWindow.npy')

	plt.figure(3)
	plt.imshow(eout,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Pair probabilities')
	plt.savefig('Figure3.png',dpi=200)

	# Overlays the pairs plot with pseudoknot plot
	plt.figure(4)
	circle=ski.morphology.disk(3)
	shadow=scipy.signal.convolve2d(pout>0.5,circle,mode='same')
	pairsout=np.multiply(eout,shadow>0)
	plt.imshow(-pairsout,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Pair probabilities over against PseudoknotSearch')
	plt.savefig('Figure4.png',dpi=200)

	plt.figure(6)
	diff=1.0*(eout<-18)-1.0*shadow
	coff=np.multiply(diff>0,eout)
	plt.imshow(-coff,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Pairings missed by partition algorithm')
	plt.savefig('Figure6.png',dpi=200)

def MakePolymerGraph():
	L1=len(LBRS1)
	L2=len(LBRS2)
	L3=len(LBRS3)

	# Load data
	eout=np.load('out_EnergyWindow.npy')

	# Break LBRS in blocks
	plt.figure(1)
	plt.imshow(-eout,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Pseudoknot energies on LBRS1-3')

	plt.figure(2)
	x=np.sum(-eout,axis=0)
	plt.plot(x)
	x1=x[0:L1]
	x2=x[L1+1:L2+L1+1]
	x3=x[L2+L1+1:]

	# Smoothen the energy plot
	filt=scipy.signal.blackman(19)
	filt=filt/np.sum(filt)
	sx1=np.convolve(filt,x1,mode='same')
	sx2=np.convolve(filt,x2,mode='same')
	sx3=np.convolve(filt,x3,mode='same')
	
	plt.plot(range(0,L1),sx1)
	plt.plot(range(L1+1,L1+L2+1),sx2)
	plt.plot(range(L1+L2+1,L1+L2+L3+1),sx3)
	
	# Obtain mins
	y1=scipy.signal.argrelmin(sx1)
	y1=y1[0]
	y2=scipy.signal.argrelmin(sx2)
	y2=y2[0]
	y3=scipy.signal.argrelmin(sx3)
	y3=y3[0]

	"""
	# Obtain max
	y1=scipy.signal.argrelmax(sx1)
	y1=y1[0]
	y2=scipy.signal.argrelmax(sx2)
	y2=y2[0]
	y3=scipy.signal.argrelmax(sx3)
	y3=y3[0]
	"""

	# Plot boundaries for LBRS chunks
	plt.plot([L1+1,L1+1],[np.min(x),np.max(x)])
	plt.plot([L1+L2+1,L1+L2+1],[np.min(x),np.max(x)])
	
	plt.plot(y1,sx1[y1])
	plt.plot(y2+L1+1,sx2[y2])
	plt.plot(y3+L1+L2+1,sx3[y3])
	plt.xlabel('base#')
	plt.ylabel('summed delta G')
	plt.title('Breaking LBRS1-3 into dynamic block sizes')
	
	# Generate dynamic blocks
	seq1=GenerateBlockDynamic(y1,LBRS1)
	seq2=GenerateBlockDynamic(y2,LBRS2)
	seq3=GenerateBlockDynamic(y3,LBRS3)
	seq=np.concatenate((seq1,seq2,seq3))
	print('Dynamic block generation complete')


	# Calculate free energies of dynamic blocks
	"""
	msize=2
	out_cx, out_pp, out_key=complexes(seq, msize,Pairs=True,Ordered=True)
	np.save('out_cx.npy',out_cx)
	np.save('out_cxpp.npy',out_pp)
	np.save('out_cxkey.npy',out_key)
	"""

	out_cx=np.load('out_cx.npy')
	out_pp=np.load('out_cxpp.npy')
	out_key=np.load('out_cxkey.npy')

	out_cxppe=BuildPPplot(out_pp,out_key,seq,out_cx)
	np.save('out_tpp.npy',out_cxppe)

	data=BuildDataList(seq,out_cx)
	SaveData('LBRS_data.txt',data)

	# Plot pseudoknot pairs energies*probability
	plt.figure(3)
	plt.imshow(-out_cxppe,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Pseudoknots from complexes function for LBRS1-3')

	# Plot pair probabilities from nupack
	plt.figure(4)
	nupk_pp=np.load('out_nupack.npy')
	plt.imshow(nupk_pp,interpolation='nearest')
	DrawL1L2L3()
	plt.colorbar()
	plt.title('Nupack predicted pair probabilities for LBRS1-3')

	# Plot block energies
	plt.figure(5)
	plt.imshow(-out_cx,interpolation='nearest')
	plt.colorbar()


	# Draw polymer graph
	plt.figure(6)
	# Get MFE structure from nupack calculations
	mfepp=getMFEPairs('749354.ocx-mfe')
	DrawPolymerGraph(out_cxppe,mfepp[13],seq)
	plt.title('Polymer graph of PseudoKnots')
	plt.xlabel('Red = LBRS1, Green = LBRS2, Cyan=LBRS3, Black=NUPACK MFE, RED = high energy (Pseudoknots)')


	plt.show()

#PseudoKnotSearch()
#OverlayPairsPlot()
MakePolymerGraph()