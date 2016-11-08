#DeltaCon: proposed in A Principled Massive-Graph Similarity Function
#node index start from 1
from __future__ import division
import pandas as pd
import numpy as np
import random
import time

from scipy.sparse import dok_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse import diags
from numpy.linalg import inv
from numpy import concatenate
from numpy import square
from numpy import array
from numpy import trace
from numpy import amax
from math import sqrt

def GenAdjacentMatrix(f):
	'''
	Get adjacent matrix from file
	'''
	data=pd.read_table(f, delimiter=' ', header=None)
	i=data[0]
	j=data[1]
	size=max([max(i), max(j)])
	adjacent=dok_matrix((size, size), dtype=np.int)
	for k in range(len(i)):
		adjacent[i[k]-1, j[k]-1]=1
		adjacent[j[k]-1, i[k]-1]=1
	for k in range(size):
		adjacent[k, k]=0
	
	return adjacent

def Partition(num, size):
	'''
	randomly divide size nodes into num groups
	'''
	partitions={}
	nodes=[x for x in range(1, size+1)]
	group_size=int(size/num)
	for i in range(num-1):
		partitions[i]=[]
		for j in range(group_size):
			node=random.choice(nodes)
			nodes.remove(node)
			partitions[i].append(node)

	#the last partition get the rest nodes
	partitions[num-1]=nodes[:]

	return partitions

def Partition2e(partitions, size):
	'''
	change partition into e vector
	size is the dimension n
	'''
	e={}
	for p in partitions:
		e[p]=[]
		for i in range(1, size+1):
			if i in partitions[p]:
				e[p].append(1.0)
			else:
				e[p].append(0.0)
	return e

def InverseMatrix(A, partitions):
        '''
        use Fast Belief Propagatioin
        CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo
        Kenneth Pao, Christos Faloutsos
        Unifying Guilt-by-Association Approaches
        return [I+a*D-c*A]^-1
        '''
	num=len(partitions)		#the number of partition

        I=identity(A.shape[0])          #identity matirx
        D=diags(sum(A).toarray(), [0])  #diagonal degree matrix

        c1=trace(D.toarray())+2
        c2=trace(square(D).toarray())-1
        h_h=sqrt((-c1+sqrt(c1*c1+4*c2))/(8*c2))

        a=4*h_h*h_h/(1-4*h_h*h_h)
        c=2*h_h/(1-4*h_h*h_h)
	
	#M=I-c*A+a*D
        #S=inv(M.toarray())

        M=c*A-a*D
	for i in range(num):
		inv=array([partitions[i][:]]).T
		mat=array([partitions[i][:]]).T
		power=1
		while amax(M.toarray())>10**(-9) and power<10:
			mat=M.dot(mat)
			inv+=mat
			power+=1
		if i==0:
			MatrixR=inv
		else:
			MatrixR=concatenate((MatrixR, array(inv)), axis=1)

	S=csc_matrix(MatrixR)
	return S

def Similarity(A1, A2, g):
        '''
        use deltacon to compute similarity
        CITATION: Danai Koutra, Joshua T. Vogelstein, Christos Faloutsos
        DELTACON: A Principled Massive-Graph Similarity Function
	g is the number of partition
        '''
	size=A1.shape[0]

	partitions=Partition(g, size)
	e=Partition2e(partitions, size)
	
        S1=InverseMatrix(A1, e)
        S2=InverseMatrix(A2, e)

        d=0
        for i in range(size):
                for j in range(g):
                        d+=(sqrt(S1[i,j])-sqrt(S2[i,j]))**2
        d=sqrt(d)
        sim=1/(1+d)
        return sim

def DeltaCon(A1, A2, g):
	#compute average sim
	Iteration=10
	average=0.0
	for i in range(Iteration):
		average+=Similarity(A1, A2, g)
	average/=Iteration
	return average

if __name__ == '__main__':
	start=time.time()
	A1=GenAdjacentMatrix('./deltacon/GRAPHS/synthetic1.txt')
        A2=GenAdjacentMatrix('./deltacon/GRAPHS/synthetic2.txt')
        sim=DeltaCon(A1, A2, 5)
        print 'sim:', sim
        end=time.time()
        print 'time:',(end-start)
