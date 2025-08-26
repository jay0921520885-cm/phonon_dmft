import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh
from itertools import product
import itertools
def build_boson_op_creation(P,Nb):   # P 為軌道數   Nb 為軌道最大佔據數
# 建立所有可能的佔據數組合
    states = list(itertools.product(range(Nb + 1), repeat=P))
# 轉成 numpy 陣列
    bhstate = np.array(states, dtype=int)  
    AH_list = [] 
    L=len(bhstate)
    hsize=(Nb+1)**P
    assert(hsize==L)
    for o in range(P):
        row = []   #storing row index for  

        col = []   #storing col index for 

        data = []  #storing data for (row, col) 
        for b in range(hsize):
           b_config=bhstate[b]  #bhstate中的小子集
           if b_config[o] < Nb :
                col.append(b)
                config_a = [] 
                #calculate the corresponding <A| 
                for i in range(P): 

                    if i == o: 

                      config_a.append(b_config[i] + 1)

                    else: 
                      config_a.append(b_config[i])

                #對比相同的陣列
                for x in range(L):
                   if np.array_equal(config_a,bhstate[x]) == True: 
                      row.append(x)  
                #算作用後的值
                val = np.sqrt(b_config[o] + 1)
                data.append(val)
            
        AH = csr_matrix((data, (row, col)), shape=(hsize, hsize), dtype=np.float64) 

        AH_list.append(AH) 
  
    return AH_list 






