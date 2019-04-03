import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np,time

from GS import *
from printer import *



import time
import os
dirOP='output_plots'
dirOD='output_data'
if dirOP not in os.listdir():
    os.mkdir(dirOP)
if dirOD not in os.listdir():
    os.mkdir(dirOD)

f = open(dirOD+'/output_problem2.txt','w')       
appendToOut(f)
print=printer
sepr="-"*80
sepe="="*96
plus="+"*96
plus3=""
for i in range(3):
    plus3+=plus+"\n"
flagT=True
class prob2(object):
    def __init__(self):
        self.df1 = pd.read_csv('mnist_train.csv',index_col=False,header=None)
        
        self.train_data = np.array(self.df1)
        self.train_label=self.train_data.T[0]
        
        self.m,self.n = self.df1.shape #OR train_mat.shape
        self.resultTask6={1: {2: 0.23, 3: 0.22, 4: 0.2095, 5: 0.1995},
                          2: {2: 0.364, 3: 0.3525, 4: 0.349, 5: 0.334},
                          4: {2: 0.556, 3: 0.528, 4: 0.5105, 5: 0.48},
                          9: {2: 0.881, 3: 0.863, 4: 0.843, 5: 0.822},
                          19: {2: 0.952, 3: 0.9365, 4: 0.9245, 5: 0.908},
                          40: {2: 0.9665, 3: 0.951, 4: 0.938, 5: 0.9235},
                          85: {2: 0.959, 3: 0.948, 4: 0.94, 5: 0.931},
                          178: {2: 0.9575, 3: 0.9475, 4: 0.937, 5: 0.9285},
                          373: {2: 0.958, 3: 0.9455, 4: 0.935, 5: 0.927},
                          783: {2: 0.96, 3: 0.946, 4: 0.9355, 5: 0.9265}}
        self.resultBonus2={1: {2: 0.258, 3: 0.263, 4: 0.2695, 5: 0.2705},
                           2: {2: 0.402, 3: 0.4095, 4: 0.418, 5: 0.4275},
                           4: {2: 0.5745, 3: 0.6015, 4: 0.62, 5: 0.6255},
                           9: {2: 0.8765, 3: 0.901, 4: 0.9015, 5: 0.902},
                           19: {2: 0.9425, 3: 0.956, 4: 0.9535, 5: 0.9565},
                           40: {2: 0.9495, 3: 0.9665, 4: 0.959, 5: 0.96},
                           81: {2: 0.904, 3: 0.93, 4: 0.9205, 5: 0.9245},
                           85: {2: 0.902, 3: 0.926, 4: 0.9135, 5: 0.9195},
                           131: {2: 0.83, 3: 0.8595, 4: 0.8395, 5: 0.842},
                           178: {2: 0.753, 3: 0.775, 4: 0.7385, 5: 0.745},
                           200: {2: 0.705, 3: 0.7365, 4: 0.6835, 5: 0.6915},
                           203: {2: 0.701, 3: 0.7335, 4: 0.6765, 5: 0.687},
                           373: {2: 0.345, 3: 0.345, 4: 0.2965, 5: 0.2955},
                           400: {2: 0.319, 3: 0.3245, 4: 0.283, 5: 0.2865},
                           783: {2: 0.3065, 3: 0.3075, 4: 0.273, 5: 0.262}}
        
        self.M=40
        self.K=2
        print("Based On previous Training:: M ={0} and K={1}".format(self.M,self.K))
        
    def updateTestData(self,file='mnist_test.csv'):
        print("Initializing/Updating Test Data")
        self.df2 = pd.read_csv(file,index_col=False,header=None)
        self.test_data= np.array(self.df2)
        self.test_label=self.test_data.T[0]
        print("Done Loading...",sepr)
        
        
    def P2_task1(self):
        mean = np.array( np.repeat([0.],self.n) )
        td = self.train_data.T
        for i in range(self.n):
            mean[i] = td[i].sum()/self.m #OR CAN USE SUM PROVIDED BY PYTHON
        self.mean=mean  
    
        td = self.train_data.T[1:].T - self.mean[1:]
        cov_mat = np.dot(td.T,td)/self.m
        self.cov_mat=cov_mat
        print("P2_TASK1:", sepr)
        print("Printing Mean:")
        if flagT:print(self.mean)
        print(sepr)
        print("Printing CoVar_Matrix:")
        if flagT:self.printArray(self.cov_mat)
        print(plus3)
        
        
    def P2_task2(self):
        print("P2_TASK2:", sepr)
       
        self.eigVal,self.eigVec=np.linalg.eigh(self.cov_mat)
        multi={}
        self.eigVec=self.eigVec.T
        count=0
        val=0
        for i in self.eigVal:
            if i not in multi:
                multi[i]=1
            else:
                multi[i]+=1
                if multi[i]==2:
                    count+=1
                    val=i
        for i in multi:
            if multi[i]>1:
                print('Eigen Value {0}  Repeating with multiplicity {1}'.format(i, multi[i]),sepe[:40])
        if count==0:
            print('None eigen values are repeating',sepe)
        flag = False
        #CHKPNT---Checkinf for all Nc2 combinations.
        for i in range(len(self.eigVec)):
            for j in range(i+1,len(self.eigVec)):
                flag = abs(np.dot(self.eigVec[i],self.eigVec[j]))>1e-12
        txt='Orthogonal' if not flag else 'Non-orthogonal'
        print('Obtained Eigen Vectors are {0}'.format(txt))
        #eigv = np.array(eigv).round(15)
        print(plus3)

    def P2_task3(self,st='gram_schimdt_input.txt'):
        print("P2_TASK3:", sepr)
       
        gs_df = pd.read_csv(st,delimiter=' ',index_col=False,header=None)
        gs_mat = np.array(gs_df)
        gs2=[]
        for i in range(len(gs_mat)):
            gs2.append(gs_mat[i][0:-1])#CHKPNT
        GS=Gram_Schmidt()
        gs_mat=np.array(gs2)
        #print(gs_mat)
        self.orth_basis = GS.gs(gs_mat)
        self.printArray(self.orth_basis)
        print(plus3)

    def P2_task4(self):
        print("P2_TASK4:", sepr)
        eig = {}
        for i in range(len(self.eigVal)):
            if self.eigVal[i] not in eig:
                eig[self.eigVal[i]] = [self.eigVec[i]]
            else:
                eig[self.eigVal[i]].append(self.eigVec[i])
        self.eigdict=dict(sorted(eig.items(),key=lambda kv: kv[0],reverse=True))#CHKPNT

    def P2_task5(self):
        print("P2_TASK5:", sepr)
       
        D = self.n-1
        low_dim,high_dim,K = 1,self.n-1,10
        r = np.exp(np.log(high_dim/low_dim)/(K-1))
        dim,err = [],[]
        #Data Centrality before PCA
        print("DATA CENTRALITY DONE BEFORE PCA")
        data_set = (np.array(self.df1.loc[:,1:])-self.mean[1:])
        self.red_data={}
        am=[81,200,203,400,131]
        for i in range(K):
            am.append(int(low_dim*r**i))
        K=len(am)
        am=sorted(am)
        print("Selecting M<DIM_REDUCTION> as {0}".format(am))
        for i in range(K):
            M = am[i]
            #print(M)
            red_data_set,rec_err = self.PCA(M,data_set)
            dim.append(M)
            err.append(rec_err)
            self.red_data[M]=red_data_set
        plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
        plt.plot(dim,err,'o-')
        plt.xlabel('Dimension')
        plt.ylabel('RMS_error')
        plt.savefig(dirOP+"/PCA-reduction-error")
        plt.show()
        self.dim=dim
        print(plus3)

    def P2_task6(self):
        print("P2_TASK6:", sepr)
        
        k=5
        batch_size=2000
        print('taking Batch_size={0} and K (of KNN)= {1}'.format(batch_size,k))
        m=self.m
        acc={}
        #from sklearn.model_selection import train_test_split  
        #from sklearn.preprocessing import StandardScaler  
        Best_K,Best_M,Best_acc=0,0,0
        
        for i in self.dim:
            
            red_data=self.red_data[i][:m-batch_size]
            red_label=self.train_label[:m-batch_size]
            test_data=self.red_data[i][m-batch_size:]
            test_label=self.train_label[m-batch_size:]
            
            tacc={}
            
            #red_data,  test_data,red_label, test_label = train_test_split(self.red_data[i],self.train_label, test_size=0.20,shuffle=False)
            
            #print(i,red_data.shape,red_label.shape,test_data.shape,test_label.shape)
            '''
            scaler = StandardScaler()  
            scaler.fit(red_data)

            red_data = scaler.transform(red_data)  
            test_data = scaler.transform(test_data)
            '''
            for j in range(2,k+1):
                start_time=time.time()
                calc_label=self.K_NN(red_data,red_label,test_data,j)
                #print(len(calc_label),j)
                count=0
                '''for (x,y) in zip(test_label,calc_label):
                    if x==y:
                        count+=1
                tacc[j]=(count/batch_size)'''
                tacc[j]=np.mean(test_label == calc_label)
                finish_time=time.time()-start_time
                print(i,j,tacc[j],finish_time,"sec")
                if(Best_acc<tacc[j]):
                    Best_acc=tacc[j]
                    Best_K=j
                    Best_M=i
            acc[i]=tacc
        self.resultTask6=acc
        print("Accuracy for various K and M")
        print(acc)
        print("Best result we get at: K={0} and M={1} with Accuracy={2}".format(Best_K,Best_M,Best_acc))
        print(plus3)
        

    def P2_bonus1(self):
        print("P2_BONUS1:", sepr)
        plt.figure(num=None, figsize=(8, 6), dpi=600, facecolor='w', edgecolor='k')
        #plt.scatter(self.red_data[2].T[0],self.red_data[2].T[1],c='r',s=1)
        for i in range(0,10):
            one=np.array([d for d,e in zip(self.red_data[2],self.train_label) if e==i])
            plt.scatter(one.T[0],one.T[1],s=1,label=i)
        plt.legend(bbox_to_anchor=(1., 1.02), loc=2,
                    borderaxespad=0.)
        
        #plt.axis('off')
        plt.savefig(dirOP+'/P2_bonus1')
        plt.show()
        print('plotting...Done!')
        print(plus3)

    def P2_bonus2(self):
        print("P2_BONUS2:", sepr)
        
        total_acc={}
        for i in self.dim:
            red_data=self.red_data[i]
            red_label=self.train_label
            t=self.bonus_knn(red_data,red_label,i)
            total_acc[i]=t
        print("Accuracy Using SKLEARN KNN",sepr)
        print( total_acc)
        print(plus3)
    
            
    def P2_Testing(self):
        print("TESTING ON GIVEN TEST DATA",sepr)
        test_data,err=self.PCA(self.M,self.test_data.T[1:].T - self.mean[1:])
        
        start_time=time.time()
        calc_label=self.K_NN(self.red_data[self.M],self.train_label,test_data,self.K)
        #print(self.red_data[self.M].shape,self.train_label.shape)
        #print("error:{0}".format(err),test_data.shape)
        #print(len(calc_label),j)
        count=0
        '''for (x,y) in zip(test_label,calc_label):
            if x==y:
                count+=1
        tacc[j]=(count/batch_size)'''
        tacc=np.mean(self.test_label == calc_label)
        finish_time=time.time()-start_time
        print("accuracy:",tacc,finish_time,"sec")
        print(plus3)
    
    def bonus_knn(self,train_data,train_label,i):
        from sklearn.model_selection import train_test_split  
        X_train, X_test, y_train, y_test = train_test_split(train_data,train_label, test_size=0.20,shuffle=False)
        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        from sklearn.preprocessing import StandardScaler  
        scaler = StandardScaler()  
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)

        from sklearn.neighbors import KNeighborsClassifier  
        classifier = KNeighborsClassifier(n_neighbors=5)  
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        
        acc = {}

        # Calculating error for K values between 1 and 40
        for i in range(2, 6):  
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            acc[i]=(np.mean(pred_i == y_test))
        print(i,acc)
        '''plt.figure(figsize=(12, 6))  
        plt.plot(range(2, 6), acc, color='red', linestyle='dashed', marker='o',  
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')  
        plt.xlabel('K Value')  
        plt.ylabel('Mean Error')  
        plt.show()'''
        return acc
            
        

    def K_NN(self,red_data,red_label,test_data,K):
        '''
        red_data  <10000,M>
        red_label <10000,1>
        test_data <1000, M>
        '''
        #print(red_data.shape,red_label.shape,test_data.shape,K)
        test_label=[]
        for i in test_data:
            diff={}
            xt=np.array(i)
            for j in range(len(red_data)):
                xq=np.array(red_data[j])
                diff[j]=np.power(np.linalg.norm(xt-xq),1)
            diff=sorted(diff.items(),key=lambda kv: kv[1])#min_distance
            count={}
            #print(diff)
            for j in range(K):
                lab=red_label[diff[j][0]]
                #print(lab)
                if lab not in count:
                    count[lab]=(1,diff[j][1])
                else:
                    count[lab]+=(count[lab][0]+1,count[lab][1])
            #CHKPNT
            count=sorted(count.items(),key=lambda kv: (kv[0],-1*kv[1]),reverse=True)#max_count
            #print(count)
            test_label.append( count[0][0])
        return test_label
            
                      
    def PCA(self,M,data):
        tm=M
        _=[]
        for i in self.eigdict:
            val=self.eigdict[i]
            if(len(val)<=tm):
                tm=tm-len(val)
                _.extend(val)
            else:
                _.extend(val[:tm])
                tm=0
                break
        U=np.array(_)# M X 784
        z=np.dot(U,data.T)# M x 784  . (10k X 784).T => M x 10k 
        recons_data=np.dot(U.T,z).T #((784 x M) . (M x 10k)).T  => (784 x 10k ).T => 10k x 784
        rec_err=np.power(np.power(np.linalg.norm(data-recons_data,axis=1),2).sum()/data.shape[0],0.5)
        #print(M,z.T.shape)
        return z.T,rec_err #  z.T => 10k x M  ; rec_err=scalar
        

        
     
    def printArray(self,A):
        for i in A:
            for j in i:
                print((j if abs(j)>1e-12 else 0.0),end=' ')# simply use round(j,10) -_-
            print()


    


p=prob2()

def gs(filename):
    print("Performing GS-->")
    if filename=="":
        p.P2_task3()
    else:
        p.P2_task3(filename)
#p.P2_task6()
def testing(filename):
    p.P2_task1()
    p.P2_task2()
    p.P2_task4()
    p.P2_task5()
    
    print('P2Task6_RESULTS -- K=2 and M=40')
    print("Performing Testing of given Test Data...")
    if filename=="":
        p.updateTestData()
    else:
        p.updateTestData(filename)
        p.P2_Testing()
    p.P2_bonus1()
    p.P2_bonus2()

#x=np.repeat(xa,4,axis=1)




import argparse        
#parser starts
parser = argparse.ArgumentParser(description='Problem 2.')
parser.add_argument('filename', metavar='<FILENAME.TXT>', type=str,
                    help='file name.txt for the problem2')


parser.add_argument('-type=gram-schimdt' , dest='func', action='store_const',
                    const=gs, default=testing,
                    help='execution of GS')


args = parser.parse_args()
args.func(args.filename)
#parser ends



f.close()

