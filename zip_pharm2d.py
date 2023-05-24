#!/home/amfmf/data/miniconda/envs/my-rdkit-env/bin/python
# coding: utf-8

# In[1]:


import rdkit
from rdkit import Chem,DataStructs
from rdkit.Chem import ChemicalFeatures, AllChem, Draw, Pharm2D,PandasTools, Descriptors
from rdkit import RDConfig
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.rdmolfiles import SDWriter
import os, time
import numpy as np
import pandas as pd
from rdkit.Chem.MolStandardize import rdMolStandardize
import gzip


# In[2]:

SLURM_ARRAY_TASK_ID = os.environ['SLURM_ARRAY_TASK_ID']

print(SLURM_ARRAY_TASK_ID)

var = os.environ['var']

print(var)

out1=var.replace(".smi",".sdf")
out2=var.replace(".smi",".sdf.gz")


t1=time.time()


# In[3]:


smi_list=['CNc1nc(I)nc2c1ncn2[C@H]1C[C@@H]([C@]2([C@@H]1C2)COP(=O)(O)O)OP(=O)(O)O',          "Cc1ccc2c(c1)C=Cc1c(C2c2cn(Cc3ccc(o3)NC(=O)c3nnn[nH]3)c(=O)[nH]c2=S)ccc(c1)C"]


# In[5]:


def tani_index2(df1, smi_list):
    df1['ROMol']=df1['Smiles'].apply(Chem.MolFromSmiles)
    mols=[Chem.MolFromSmiles(i) for i in smi_list]
    
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=2)
    sigFactory.SetBins([(0,3),(3,6),(6,10)])
    sigFactory.skipFeats=[ 'PosIonizable','NegIonizable','ZnBinder',]
    sigFactory.Init()
    sigFactory.GetSigSize()
    
    hit_mols=[]
    for mol in mols:
        hit_fp=Generate.Gen2DFingerprint(mol,sigFactory)
        print(f"len of hit_fp:{len(hit_fp)}")
        
        hit_keys = hit_fp.GetOnBits()
        print(f"getOn bits:{len(hit_keys)}")

        bit_v=[0]*45

        for i in hit_keys:
            bit_v[i]=1
        print(f"len of bit_vec:{len(bit_v)}")

        hit_v=DataStructs.cDataStructs.CreateFromBitString(str(bit_v))
        print(f"len of hit_vec:{len(hit_v)}")
        hit_mols.append(hit_v)
    
    
    print(f"count of hits:{len(hit_mols)}")
    
    fps=[]
    for mol in df1['ROMol']:
        fp1 = Generate.Gen2DFingerprint(mol,sigFactory)
        fp1_keys = fp1.GetOnBits()
        init_list = [0]*45
    
        for fp1_key in fp1_keys:
            init_list[fp1_key] = 1
        fp =init_list 
        
        ex=DataStructs.cDataStructs.CreateFromBitString(str(fp))
        fps.append(ex)
    
    print(f"count of fps:{len(fps)}")
     
    tani1=DataStructs.BulkTanimotoSimilarity(hit_mols[0], fps)
    tani2=DataStructs.BulkTanimotoSimilarity(hit_mols[1], fps)
        
    df1['tanimoto_1']=tani1
    df1['tanimoto_2']=tani2
    
    df2=df1[df1['tanimoto_1']> 0.6]
    df2.drop(columns=['tanimoto_2'], inplace=True, index=None)
    df2.reset_index(inplace=True)
    df3=df1[df1['tanimoto_2']> 0.6]
    df3.drop(columns=['tanimoto_1'], inplace=True, index=None)
    df3.reset_index(inplace=True)
    df4=pd.concat([df2, df3], axis=0)
    df4.reset_index(inplace=True)
    df4.drop(columns='index')
    df4.drop_duplicates(subset='Title', keep='first', inplace=True)
    x=int(df1.shape[0]*0.012)
    y=int(df4.shape[0])
    df4.fillna(0)
    if y<x:
        None
    elif y>=x:
        col_list=['tanimoto_1', 'tanimoto_2']
    
        df4['tani']=df4[col_list].sum(axis=1)
   
        df4=df4.nlargest(x, ['tani'])
    
    print(f"final df shape:{df4.shape}")
    
    
    return df4
    


# In[ ]:





# In[6]:


df11=pd.read_csv(var, sep=' ', names=['Smiles',"Title"])


# In[7]:


df1=tani_index2(df11, smi_list)

print(df1.shape)

# In[8]:


mol_wt=[]
for mol in df1['ROMol']:
     mol_wt.append(Descriptors.MolWt(mol))



df1['mol_wt']=mol_wt

df1=df1[df1['mol_wt'] >250]
print(f"mol_wt filter:{df1.shape}")


def aliphatic_atoms(mol): #Gives the generator of list of aliphatic chains if present in each compounds.
    rot_atom_pairs = list(mol.GetSubstructMatches(Chem.MolFromSmarts("[R0;D2]")))
    l=[list(x) for x in rot_atom_pairs]
    f=[(i[0]) for i in l]
    import itertools
    for i, j in itertools.groupby(enumerate(f), lambda x: x[1] - x[0]):
        j = list(j)
        start = j[0][1]
        length = len(j)
        if length == 1:
            yield (start-start)
        else:
            yield ((start+length)-start)

def connect_aa(df):#connects to df
    ali_atoms=[]
    for mol in df['ROMol']:
        ali_atoms.append(list(aliphatic_atoms(mol)))    
    
    return ali_atoms

def Aliphatic_c(ali_atoms): #Selects the longest sidechain amongst the list of sidechains.
    ali_c=[]
    for i in ali_atoms:
        if i and max(i):
            ali_c.append(max(i))
        else:
            ali_c.append(0)
    return ali_c

def aliphatic_atom_count(df):#overall program to count the longest aliphatic chain in a given molecule.
    result=connect_aa(df)
    ali_c= Aliphatic_c(list(result))
    df['Aliphatic_chain_len']=ali_c
    return df

df1=aliphatic_atom_count(df1)


df1=df1[df1['Aliphatic_chain_len']<=6]
print(f"After ali_atom_filter:{df1.shape}")

def chiral_center_and_ringcount(df):
    y=[]
    z=[]
    for mol in df['ROMol']:
        chirals_c=len(Chem.FindMolChiralCenters(mol,force=True,includeUnassigned=False,useLegacyImplementation=True))
        num_rings= Chem.GetSSSR(mol)
        y.append(chirals_c)
        z.append(num_rings)
    
    df['chiral_c']=y
    df['ring_c']=z      
    
    return df

chiral_center_and_ringcount(df1)
df1=df1[df1['chiral_c']<3]
print(f"After chiral center filter:{df1.shape}")



def count_four_fused_rings(mol): #Gives the count of fused ring system >=4 as 1 
    ri = mol.GetRingInfo()    
    d=[]  
    for ring in ri.AtomRings():
        d.append(set(ring))                
    total_rings=len(d)
    fused_ring_count=0
       
    from itertools import combinations
    f=[]
    for items in list(combinations(d, 4)):
        if items[0].intersection(items[1]):
            if items[1].intersection(items[2]):
                if items[2].intersection(items[3]):
                    f.append(items[0].union(items[1], items[2], items[3]))
                    fused_ring_count=+1
                        
    return  fused_ring_count

def four_fusedring_count(df): #applying the above function on df
    b=[]
    for mol in df['ROMol']:
        c=count_four_fused_rings(mol)
        b.append(c)
    df['count_four_fused_rings']=b 
    return df
        
df1=four_fusedring_count(df1)


df1=df1[df1['count_four_fused_rings']<1]

print(f"count after fused ring filter:{df1.shape}")

def car_acids(df):
    p=Chem.MolFromSmarts('[CX3](=O)[OX1H0-,OX2H1]')
    y=[]
    for i in df['ROMol']:
        y.append(len(i.GetSubstructMatches(p)))
    df['carboxyl_group_count']=y
    return df

def aliphatic_amino_count(mol): #Aliphatic amino group count
    atoms=[x for x in mol.GetAtoms()] 
    ind=[x.GetIdx() for x in atoms]
    atom_num=[x.GetAtomicNum() for x in atoms]
    atom_hyb=[x.GetHybridization() for x in atoms]
    comb=list(zip(atom_num, atom_hyb))
    Natoms=[(i[1]) for _,i in enumerate(comb) if i[0]==7]
    count=len([x for x in Natoms if x == rdkit.Chem.rdchem.HybridizationType.SP3])
    comb=list(zip(ind,atom_num))
    a=[i for i,j in comb if j==7]
    b=[i for i,j in comb if j==16]
    c=[[i+1, i-1, i+2, i-2, i+3, i-3, i+4, i-4, i+5, i-5] for i,j in comb if j==7]
    m=[]
    n=[]
    for i in b:
        for j in c:
            for k in j:
                if i==k:
                    m.append(i)
                   
                else:
                    None
    count=max(count-len(m),0)
   
    return count

df1['ali_N']=df1['ROMol'].apply(aliphatic_amino_count)
df1=car_acids(df1)

df1=df1[(df1['ali_N']<=2)] #filter for  Number of carboxyl groups + Number of amino N (with aliphatic neighbors) <= 1

df1=df1[(df1['carboxyl_group_count']<=1)] #carboxyl count (2,1)(1,1)(0,1)(2,0)(1,0)(0,0)

df2=df1.drop(df1.loc[(df1['carboxyl_group_count']==1)& (df1['ali_N'].isin([1,2]))].index)
df2=df2[['Smiles', 'Title', 'ROMol', 'tanimoto_1','tanimoto_2', 'tani']]
print(f"final_shape:{df2.shape}")


# In[9]:


def standardize(mol): 
    
     
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol) 
     
    # if many fragments, get the "parent" (the actual mol we are interested in) 
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
         
    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    
    
    return uncharged_parent_clean_mol
 

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

def protonate_ali_amino(mol):#charging sp3 Nitrogens with +1 
    for at in mol.GetAtoms(): 
            if at.GetAtomicNum() == 7 and at.GetHybridization()==rdkit.Chem.rdchem.HybridizationType.SP3 and at.GetFormalCharge()==0:
                at.SetFormalCharge(1)
    return mol

def deprotonation_cooh(mol):
    deprotonate_cooh  =  AllChem.ReactionFromSmarts("[C:1](=[O:2])-[OH1:3]>>[C:1](=[O:2])-[O-H0:3]")
    
    m_deprot  =  deprotonate_cooh.RunReactants((mol,))
        
    return m_deprot[0][0]  if  m_deprot  else  mol 


# In[10]:


def main(df2):
    li=[]
    for mol in df2['ROMol']:
        f=standardize(mol)
        f.UpdatePropertyCache()
        g=neutralize_atoms(f)
        h=protonate_ali_amino(g)
        d=deprotonation_cooh(h) 
        Chem.SanitizeMol(d)
        m=Chem.AddHs(d)
        AllChem.EmbedMolecule(m)
        li.append(m)
    df2['new_mols']=li
    return df2
    
    


# In[11]:


df2=main(df2)


# In[12]:


PandasTools.WriteSDF(df2, out1, molColName='new_mols', idName='Title', properties=list(df2.columns)) #out1=var.replace(".smi",".sdf")


# In[13]:


fp = open(out1,"rb") #out2=var.replace(".smi",".sdf.gz")
data = fp.read()
bindata = bytearray(data)
with gzip.open(out2, "wb") as f:
    f.write(bindata)



t2=time.time()


# In[15]:


print(f"time is:{t2-t1:.2f}")


# In[ ]:



