import numpy as np
class Gram_Schmidt(object):
	def projua(self,u,a):
		u=np.array(u)
		a=np.array(a)
		proj= (np.dot(u,a)/np.dot(u,u)) * u
		return proj
	
	def gs(self,A):
		A=np.array(A,dtype='float')
		B=np.array(A,dtype='float')
		U=[]
		U.append(B[0])
		for i in range(1,len(A)):
			tmp=np.array(B[i])
			for j in range(0,i):
				tmp-=self.projua(U[j],B[i])

			U.append(tmp)						
		return np.array(U) #Only Gram Based on Output Given in the Problem statement.
		#return self.normalize(np.array(U).T)#NORMALIZE
	
	def normalize(self,A):
		Q=np.array(A,dtype='float')
		for j in range(len(A[0])):
			U=0
			for i in range(len(A)):
				#print(A[i][j])
				U+= A[i][j]*A[i][j]
				#print(A[i][j],U)
			#print(U)
			U=U**(0.5)
			#print(U)
			for i in range(len(A)):
				Q[i][j]=A[i][j]/U
		return Q.T
