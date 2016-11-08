#DeltaCon0 proposed in A Principled Massive-Graph Similarity Function
from __future__ import division
import pandas as pd
import numpy as np
import time

from scipy.sparse import dok_matrix
from scipy.sparse import identity
from scipy.sparse import diags
from numpy.linalg import inv
from numpy import square
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

def Similarity(A1, A2):
	'''
	use deltacon0 to compute similarity
	CITATION: Danai Koutra, Joshua T. Vogelstein, Christos Faloutsos
	DELTACON: A Principled Massive-Graph Similarity Function
	'''
	S1=InverseMatrix(A1)
	S2=InverseMatrix(A2)
	
	d=0
	for i in range(A1.shape[0]):
		for j in range(A1.shape[0]):
			d+=(sqrt(S1[i,j])-sqrt(S2[i,j]))**2
	d=sqrt(d)
	sim=1/(1+d)
	return sim	

def InverseMatrix(A):
	'''
	use Fast Belief Propagatioin
	CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo
	Kenneth Pao, Christos Faloutsos
	Unifying Guilt-by-Association Approaches
	return [I+a*D-c*A]^-1
	'''
	I=identity(A.shape[0])		#identity matirx
	D=diags(sum(A).toarray(), [0])	#diagonal degree matrix

	c1=trace(D.toarray())+2
	c2=trace(square(D).toarray())-1
	h_h=sqrt((-c1+sqrt(c1*c1+4*c2))/(8*c2))

	a=4*h_h*h_h/(1-4*h_h*h_h)
	c=2*h_h/(1-4*h_h*h_h)

	
	#M=I-c*A+a*D
	#S=inv(M.toarray())
	'''
	compute the inverse of matrix [I+a*D-c*A]
	use the method propose in Unifying Guilt-by-Association equation 5
	'''	
	M=c*A-a*D
	S=I
	mat=M
	power=1
	while amax(M.toarray())>10**(-9) and power<7:
		S=S+mat
		mat=mat*M
		power+=1

	return S

if __name__ == '__main__':
	start=time.time()
	A1=GenAdjacentMatrix('./deltacon/GRAPHS/synthetic1.txt')
	A2=GenAdjacentMatrix('./deltacon/GRAPHS/synthetic2.txt')
	sim=Similarity(A1, A2)
	print 'sim:', sim
	end=time.time()
	print 'time:',(end-start)
