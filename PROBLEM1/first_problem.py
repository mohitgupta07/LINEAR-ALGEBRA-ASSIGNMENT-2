import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from eigenVector import *
from centralities import *
from second_problem import *
from printer import *
from sklearn.cluster import KMeans

import os
dirOP='output_plots'
dirOD='output_data'
if dirOP not in os.listdir():
	os.mkdir(dirOP)
if dirOD not in os.listdir():
	os.mkdir(dirOD)

f = open(dirOD+'/output_problem1.txt','w')       
appendToOut(f)
print=printer
sepr="-"*80
sepe="="*96
plus="+"*96
plus3=""
for i in range(3):
	plus3+=plus+"\n"


class Graph(object):
	def __init__(self,path='got_network.gml',load=True):
		self.path=path
		#self.graph = nx.Graph()
		assert path!=""
		if load:
			self.loadGML()
			self.size=len(self.graph.nodes)
	def loadGML(self):
		self.graph=nx.read_gml(self.path)
		print("Graph loaded",self.graph)
	def showGraph(self):
		nx.draw(self.graph)
		plt.show()
	def loadSample(self):
		#self.graph=nx.dodecahedral_graph()
		self.graph=nx.gnp_random_graph(20,0.5)
	def showNodeDegDis(self):#Nodes on x-axis and their degree
		g=self.graph
		degseq=g.degree()
		print("degree distribution of nodes",degseq)
		plt.loglog(degseq,'b-',marker='o')
		plt.title("Degree rank plot")
		plt.ylabel("degree")
		plt.xlabel("rank")
		plt.show()


	def plotAns(self,title,xlabel,ylabel,data,output):
		plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
		_ = list(dict(data).values())
		BUCKETS = max(_)+1
		y=plt.hist(_,bins=BUCKETS,range=(0,BUCKETS),align='left',rwidth=0.5)		
		plt.axis([min(y[1]), max(y[1]), min(y[0]), max(y[0])])
		plt.title(title)
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)
		plt.savefig(dirOP+"/"+output)
		plt.show()

	def plotAnsNX(self,title,xlabel,ylabel,data,output):
		plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
		_ = list(dict(data).values())
		y=plt.hist(_,align='left',rwidth=0.5)		
		plt.axis([min(y[1]), max(y[1]), min(y[0]), max(y[0])])
		plt.title(title)
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)
		plt.savefig(dirOP+"/"+output)
		plt.show()

	def showDegCountDis(self):#Distribution based on nodes per degree vs degree.
		g=self.graph
		degseq=g.degree()
		self.plotAns(title="Problem1task1",
					 xlabel="Degree",ylabel="Nodes per degree",
					 data=degseq,
					 output="problem_1_task_1.png")
		'''degcnt={}
		for v,d in degseq:
			if d not in degcnt:
				degcnt[d]=1
			else:
				degcnt[d]+=1
		key=sorted(degcnt.keys())
		
		X,Y=[],[]
		for k in key:
			X.append(k)
			Y.append(degcnt[k])
		'''
	def getAdjMat(self):
		edges=self.graph.edges
		A=np.zeros((self.size,self.size))
		for a,b in edges:
			a=int(a)-1
			b=int(b)-1
			A[a][b]+=1
			A[b][a]+=1
		print(A)
		#A=nx.to_numpy_matrix(self.graph)
		#print(A)
		return A
	def getEigenValues(self,A):
		A=np.array(A)# TYPE CASTING
		assert A.shape[0]==A.shape[1] #Square matrix
		lambd=[1.]
		A2=np.array(A)
		n=A.shape[0]
		for k in range(1,n+1):
			a2= - (A2.trace()/k)
			lambd.append(a2)
			A2+= np.diag(np.repeat(a2,n))
			A2 = np.dot(A,A2)
		#lambd[-1]=0
		lambd=sorted(np.roots(lambd),reverse=True)
		flagReal=True
		for lam in lambd:
			if(lam.imag != 0.):
				flagReal=False
				break

		print('Eigen Values Generated',sepe)	
		print("The eigen values are {0}".format( "Real" if flagReal else "Complex"))
		
		vectors=[]
		for lam in lambd:
			X=self.getEigenVector(A,lam,n,1e-12)
			X=np.array(X)
			vectors.append(X)
			ans1= float(lam)*X# lambda.X
			ans2= np.dot(A,X)# A.X 
			error=np.linalg.norm( ans1-ans2)#Ax- lambda.x==0
			print('For given lambda(approx)={0}, we get error of L2_norm[( A-lambda).X - 0] ={1}'.format(lam,error))
		print('Generated Eigen Vectors',sepe)
		self.eigvectors,self.lambd=vectors,lambd
		return lambd,vectors
	def getEigenVector(self,A,lambd,n,error):
		Ax=A-np.diag(np.repeat(lambd+error,n))
		#c=np.zeros( (len(A),1))
		Ap=np.array(Ax)
		X=self.invPow(Ap,itr=10)
		return X
	def P1_task1(self):
		print("P1_task1",sepr)
		print("Plotting::::")
		self.showDegCountDis()
		print(plus3)
	def P1_task2(self):
		print("P1_task2",sepr)
		self.c=centralities(self.graph)
		self.c.getCentralities()
		x=self.c.vertex_freq
		x = sorted(x.items(), key=lambda kv: kv[1])
		names=self.graph.nodes(data='name')
		self.names=dict(names)
		print("Top two central nodes",sepr)
		print(names[x[-1][0]],names[x[-2][0]])
		
		self.plotAns(title="Problem1task2-Node centrality",
					 xlabel="Nodes",ylabel="Count",
					 data=x,
					 output="problem_1_task_2.png")
		print('Using networkX library to plot- closeness centrality')
		x= nx.closeness_centrality(self.graph)
		
		self.plotAnsNX(title="Problem1task2-Node centrality",
					 xlabel="Nodes",ylabel="Count",
					 data=x,
					 output="problem_1_task_2_Networkx.png")
		
		print(plus3)
		
	def P1_task3(self):
		print("P1_task3",sepr)
		
		#degCentral= nx.edge_betweenness_centrality(self.graph)
		
		x=self.c.edge_freq
		sorted_by_value = sorted(x.items(), key=lambda kv: kv[1])
		self.cEdge=sorted_by_value[-1]
		print(sorted_by_value[-1])
		print("Edge between {0} and {1} is the central one".format( self.names[sorted_by_value[-1][0][0]],self.names[sorted_by_value[-1][0][1]]))
		
		self.plotAns(title="Problem1task3-Edge Betweeness centrality",
					 xlabel="edges centrality",ylabel=" no. of edges",
					 data=x,
					 output="problem_1_task_3.png")
		print('Using networkX library to plot - Edge betweeness centrality')
		
		x= nx.edge_betweenness_centrality(self.graph)
		self.plotAnsNX(title="Problem1task3-Edge Betweeness centrality",
					 xlabel="edges centrality",ylabel=" no. of edges",
					 data=x,
					 output="problem_1_task_3_NetworkX.png")
		
		
		print(plus3)
		
	def P1_task4(self):
		#get adj matrix
		print('P1_task4',sepr)
		print('Adjancy Matrix:')
		A=self.getAdjMat()#Adj matrix
		n=len(self.graph.nodes)
		#D=np.zeros( (n,n) , dtype='int32')#degree matrix
		x=self.graph.degree()
		x=dict(x)
		D=np.diag(list(x.values()))
		'''for i in range(n):
			D[i][i]=self.graph.degree()[str(i)]
		'''
		L=D-A#Laplace Matrix
		#print('Laplacian Matrix \n',L)
		#print('np lap\n',nx.laplacian_matrix(self.graph))
		Dhalf=np.sqrt(D)
		Dihalf=np.zeros( (n,n))
		for i in range(n):
			if Dhalf[i][i]!=0:
				Dihalf[i][i]=1/Dhalf[i][i] 
		X= np.matmul(Dihalf,L)
		#print(Dihalf,'\n\n\n\n\n', Dhalf)
		NL= np.matmul(X,Dhalf)#Normalized Laplace matrix
		self.L=np.array(L)
		print('Laplacian Matrix==')
		print(self.L)
		print('Normalized Laplacian Matrix==')
		print(NL)
		print(sepe)
		self.getEigenValues(np.array(L))
		print(plus3)
		
	def P1_task5(self):
		print('P1_TASK5',sepr)
		#smallest two eigen values/vectors
		print("smallest one:",self.lambd[-1],self.eigvectors[-1])
		print("second smallest one:",self.lambd[-2],self.eigvectors[-2])
		#report...
		val=self.lambd[-1]
		error=val-float(0)
		vector=self.eigvectors[-1]
		one=vector[0]
		vector=[v/one for v in vector]
		errorV=np.linalg.norm(np.array(vector)-np.array([1]*self.size))
		print("difference in eigen value(original/calculated/error):",0,val,error)
		print("difference in eigen vector(original/calculated/error):",[1]*self.size,"\n",vector,"\n",errorV)
		print(plus3)
	def P1_task6(self):
		print('P1_task6',sepr)
		vector=self.eigvectors[-2]
		nodes=list(self.graph.nodes)
		self.nodes=nodes
		A = [x for x in nodes if vector[int(x)-1]>=0]
		B = [x for x in nodes if vector[int(x)-1]<0]
		print(A)
		print(B)
		values=[[1,0.6,0.3]]*self.size
		for i in A:
			values[int(i)-1]=[0.3,0.6,1]
		names=self.graph.nodes(data='name')
		names=dict(names)
		print('Showing the Graph')
		edgeval=[[0.8,0.8,0.8]]*len(self.graph.edges)
		edges=list(self.graph.edges)
		ce=self.cEdge[0]
		for i in range(len(edges)):
			if edges[i][0]==ce[0] and edges[i][1]==ce[1]:
				edgeval[i]=[0.3,0.3,0.3]
				xx=values[int(edges[i][0])-1]
				values[int(edges[i][0])-1]=[xx[0],xx[1]+0.1,xx[2]]	
				xx=values[int(edges[i][1])-1]
				values[int(edges[i][1])-1]=[xx[0],xx[1]+0.1,xx[2]]
		plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
		nx.draw(self.graph,node_color=values,edge_color=edgeval,with_labels=True,labels=names)#,cmap=plt.get_cmap('jet')
		plt.savefig("output_plots/Problem_1_task_6_clustering")
		plt.show()
		self.grpA=A
		self.grpB=B
		self.names=names
		print('plotting done')
		print('Using K means clustering for the given problem',sepe)
		self.kmeans()
		print(plus3)

	def P1_task7(self):
		print('P1_task7',sepr)
		
		degree=self.graph.degree()
		degree=dict(degree)
		degree_dist_A = [degree[i] for i in self.grpA]
		degree_dist_B = [degree[i] for i in self.grpB]
		var_1 = np.var(degree_dist_A)/np.mean(np.var(degree_dist_A))
		var_2 = np.var(degree_dist_B)/np.mean(np.var(degree_dist_B))
		if var_1>=var_2:
			print('Be friend with ',[self.names[i] for i in self.grpA])
		else:
			print('Be friend with ',[self.names[i] for i in self.grpB])
		print(plus3)
	def P1_bonus1(self):
		print('1_bonus1',sepr)
		
		eigv,eigV = np.linalg.eigh(self.L)
		eigV = eigV.T
		self.np_eigV=eigV
		self.np_eigv=eigv
		partition = eigV[list(eigv).index(sorted(eigv)[1])]
		#print(partition)
		A = [x for x in self.nodes if partition[int(x)-1]>=0]
		B = [x for x in self.nodes if partition[int(x)-1]<0]
		print("Group A:",A)
		print("Group B:",B)
		print('plotting....',sepe)
	

		values=[[1,0.6,0.3]]*self.size
		for i in A:
			values[int(i)-1]=[0.3,0.6,1]
		names=self.graph.nodes(data='name')
		names=dict(names)
		edgeval=[[0.8,0.8,0.8]]*len(self.graph.edges)
		edges=list(self.graph.edges)
		ce=self.cEdge[0]
		for i in range(len(edges)):
			if edges[i][0]==ce[0] and edges[i][1]==ce[1]:
				edgeval[i]=[0.3,0.3,0.3]
				xx=values[int(edges[i][0])-1]
				values[int(edges[i][0])-1]=[xx[0],xx[1]+0.1,xx[2]]	
				xx=values[int(edges[i][1])-1]
				values[int(edges[i][1])-1]=[xx[0],xx[1]+0.1,xx[2]]
		plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
		nx.draw(self.graph,node_color=values,edge_color=edgeval,with_labels=True,labels=names)#,cmap=plt.get_cmap('jet')
	


		plt.savefig(dirOP+"/Problem_1_bonus_1_clustering")
		plt.show()
		print('done')
		print(plus3)

	def P1_bonus2(self):
		pass
	def P1_bonus3(self):
		print('P1_bonus3',sepr)
		eigV=self.np_eigV
		eigv=self.np_eigv
		partition = eigV[list(eigv).index(sorted(eigv)[-2])]
		#print(partition)
		A = [x for x in self.nodes if partition[int(x)-1]>=0]
		B = [x for x in self.nodes if partition[int(x)-1]<0]
		print("Group A:",A)
		print("Group B:",B)
		values=[[1,0.6,0.3]]*self.size
		for i in A:
			values[int(i)-1]=[0.3,0.6,1]
		names=self.graph.nodes(data='name')
		names=dict(names)
		edgeval=[[0.8,0.8,0.8]]*len(self.graph.edges)
		edges=list(self.graph.edges)
		ce=self.cEdge[0]
		for i in range(len(edges)):
			if edges[i][0]==ce[0] and edges[i][1]==ce[1]:
				edgeval[i]=[0.3,0.3,0.3]
				xx=values[int(edges[i][0])-1]
				values[int(edges[i][0])-1]=[xx[0],xx[1]+0.1,xx[2]]	
				xx=values[int(edges[i][1])-1]
				values[int(edges[i][1])-1]=[xx[0],xx[1]+0.1,xx[2]]
		plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
		nx.draw(self.graph,node_color=values,edge_color=edgeval,with_labels=True,labels=names)#,cmap=plt.get_cmap('jet')
		plt.savefig(dirOP+"/Problem_1_bonus_3_clustering")
		plt.show()
		
	def P1_bonus4(self):
		pass
	
	def checkDiffinEV(self):#diff between original numpy result vs mine calc result.
		cev=self.eigvectors[::-1]#calculated eigen vector
		for i in range(self.size):
			print(np.linalg.norm(abs(self.np_eigV[i])- abs(cev[i])))

	def kmeans(self):
		adjacency_matrix = self.getAdjMat()
		nodes = list(self.graph.nodes)
		kmean_algo = KMeans(n_clusters=2, n_init=200)
		kmean_algo.fit(adjacency_matrix)
		#     print(kmean_algo.labels_)
		A = [x for x in nodes if kmean_algo.labels_[int(x) - 1] == 1]
		#     print(array_1)
		names = self.names
		B = [x for x in nodes if kmean_algo.labels_[int(x) - 1] == 0]
		print("Group A:",A)
		print("Group B:",B)
		values=[[1,0.6,0.3]]*self.size
		for i in A:
			values[int(i)-1]=[0.3,0.6,1]
		names=self.graph.nodes(data='name')
		names=dict(names)
		edgeval=[[0.8,0.8,0.8]]*len(self.graph.edges)
		edges=list(self.graph.edges)
		ce=self.cEdge[0]
		for i in range(len(edges)):
			if edges[i][0]==ce[0] and edges[i][1]==ce[1]:
				edgeval[i]=[0.3,0.3,0.3]
				xx=values[int(edges[i][0])-1]
				values[int(edges[i][0])-1]=[xx[0],xx[1]+0.1,xx[2]]	
				xx=values[int(edges[i][1])-1]
				values[int(edges[i][1])-1]=[xx[0],xx[1]+0.1,xx[2]]
		plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
		nx.draw(self.graph,node_color=values,edge_color=edgeval,with_labels=True,labels=names)#,cmap=plt.get_cmap('jet')
		plt.savefig(dirOP+"/Problem_1_TASK6_KMEANS")
		plt.show()

	def qrMethod(self,A):
		print('running qr metthod')
		EV=np.identity(len(A))
		A=np.array(A,dtype='float')	
		for i in range(30):
			U=self.GS(A)
			Q=self.normalize(U)
			R=np.array(np.dot(Q.T,A) ,dtype='float')
			Ak=np.dot(R,Q)
			A=Ak# Ak+1=Rk.Qk
			EV=np.dot(Q,EV)
		#return Ak,Q,R
		print (EV)
		self.printer(A)
	def printer(self,A):
		for i in range(len(A)):
			for j in range(len(A[0])):
				if(i==j):
					print(A[i][j],end='\t')
			print()
		print('\n\n\n')
	def projua(self,u,a):
		u=np.array(u)
		a=np.array(a)
		
		proj= (np.dot(u,a)/np.dot(u,u)) * u
		return proj
	def GS(self,A):
		A=np.array(A,dtype='float')
		B=np.array(A.T,dtype='float')
		U=[]
		U.append(B[0])
		for i in range(1,len(A[0])):
			tmp=np.array(B[i])
			for j in range(0,i):
				tmp-=self.projua(U[j],B[i])

			U.append(tmp)						
		
		return np.array(U).T
	def invPow(self,A,itr=50):
		A=np.array(A)
		#Ai=np.linalg.inv(A)	#Better accuracy...More precision...Less error.	
		Ai=solver(A,len(A),len(A))#Inverse from second_problem of 1st assignment.
		b=np.array( [np.random.random_integers(5,100) for i in range(len(A))])
		for i in range(itr):
			#print('itr',i)
			bk= np.dot(Ai,b)
			C=np.linalg.norm(bk)#L2 norm.
			b=bk/C
			if (b==bk).all():
				break
		return b
			
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
		return Q
	def tryQR(self,A):
		self.qrMethod(A)
		
	def dummy(self):
		g=nx.Graph()
		g.add_node('1')
		g.add_node('2')
		g.add_node('3')
		g.add_edge('1','2')
		g.add_edge('2','3')
		return g


def call(file):
	g=Graph(file)
	g.P1_task1()
	g.P1_task2()
	g.P1_task3()
	g.P1_task4()
	g.P1_task5()
	g.P1_task6()
	g.P1_task7()
	g.P1_bonus1()
	g.P1_bonus3()
	
#call('aa')

import argparse        
#parser starts
parser = argparse.ArgumentParser(description='Problem 1')
parser.add_argument('filename', metavar='<FILENAME.TXT>', type=str,
			help='file name.txt for the problem1')


parser.add_argument(  dest='func', action='store_const',
			const=call,
			help='execution of Problem1')


args = parser.parse_args()
args.func(args.filename)
#parser ends




		
f.close()
