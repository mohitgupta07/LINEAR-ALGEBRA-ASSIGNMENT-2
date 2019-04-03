prc=8
prcx=12
def solver(A,n,m):#A is augmented Matrix, can simply make m=m+1
    #Step1:- Reduce to Upper Triangler matrix <echolen form> Also memorizing each operation
    #print(A.shape,n,m)
    Operations,K,I,free,basic,pivot,flag=[],-1,-1,[],[],[],0
    while(K<m and I<(n-1)):#here k<m is correct as k==m is B matrix so it's good.
        I+=1#row for pivot...
        K+=1#Column at which pivot will occur ...worst case k==m
        #if(I==n): either I< n-1 in while loop or use this and make I<n
        #    break
        flag=0
        print('mc',A[I][K])
        while( A[I][K]==0):
            flag=0
            for i in range(I+1,n):                
                if(A[i][K]!=0):#checking for non-zero column for pivot possibility
                    Operations.append(['SWITCH',I,i])
                    flag=1
                    tmp=list(A[I])
                    A[I]=A[i]#perform switch
                    A[i]=tmp
                    print(A[i]==A[I])
                    break
                #end of for loop...
            if(flag==0):#add free variable and move on.
                free.append(K)
                print(K)
                K+=1
            if(K>m or I>=n):#I>=n is redundant here. But leave it for now.
                #Ops... No more pivot left.
                flag=2#since I dont know how to add label so better use flag to get out.
                break
        pass#end of while loop.
        if(flag==2):
            break#CRAP! need to exit the loop...
        #We now have a non-zero column entry so add a basic variable to our list
        basic.append(K)
        pivot.append((I,K))
        #Converting the row below the pivot K to 0.
        scale=1/float(A[I][K])
        Operations.append(['MULTIPLY',scale,I])#adding to operations list.
        A[I]=SCALE(A,I,scale)#scaling done...returns a row
        #print(A)
        A,ops=ConvertRowsToZero(A,I,K,n)
        if(len(ops)!=0):#sometimes zero operations can occur
            Operations.append(ops)
    pass#end of outer while loop
    #Add those free variables which were unreachable in step1 and couldnt get added.
    for i in range(basic[-1]+1,m):
        free.append(i)# add i to free list...       
    #Step2:-Convert into identity matrix if possible..simply reduced echolen form...
    #to do so make each column zero which has a basic variable(pivot) in it and obviously except the pivot.
    for i,k in pivot:
        A,ops=MakeItReduced(A,i,k)
        if(len(ops)!=0):#sometimes zero operations can occur
            Operations.append(ops)
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j]=round(A[i][j],prc)
    
    if m in free:
        free.remove(m)     
    print("matrix",free,basic,pivot,A)
    
    #Step3:-Get Free variables:THis task is required to be done for question 1.
    isIn=isInconsistent(basic,m)#returns true if inconsistent else false
    maxLimitCrossed=False
    free=list(set(free))
    if(not(isIn)):
        equations=CreateX(A,free,pivot,n,m)
        X=[5 for i in range(0,m)]
        maxfree=1
        #print(free)
        #print(free)

        xf=len(free)-1#current incrementer.
        for _ in range(0,int(maxfree)):
            #print(_,xf,free[xf],m, free ,"previous X: ", X)
            for i in equations:
                exec(i)
            for i in range(0,m):
                if isinstance(X[i],int):
                    X[i]=round(X[i])
                if isinstance(X[i],float):
                    X[i]=round(X[i],prc)
            
            
                    
        if(not(maxLimitCrossed)):
    
            if(len(free)==0):
                sx=' '.join(map(str,X[0:m]))
                print(sx)
               
            else:#inf soln...
                sx=' '.join(map(str,X[0:m]))
                sf='free Variables={'
                for fr in free:
                    sf+=' X['+str(fr)+'],'
                sf+='}'
                sb=sf+' Equations={'
                sb+=' ; '.join(map(str,equations[::-1]))
                sb+='}'
    return X
def SCALE(A,I,scale):
    return [i*round(scale,prc) for i in A[I]]
def ConvertRowsToZero(A,I,K,n):
    ops=[]
    for i in range(I+1,n):
        scale=-1*float(A[i][K])
        if(scale==0):
            continue
        ops.append(['MULTIPLY&ADD',scale,I,i])
        tmpI=SCALE(A,I,scale)#getting scaled value in tmpI.
        A[i]=[round(x+y,prcx) for x,y in zip(tmpI,A[i])]#element-wise addition. Thanks Python!
    return A,ops        
def MakeItReduced(A,I,K):
    ops=[]
    for i in range(0,I):
        #print(A[i])
        scale=-1*float(A[i][K])
        if(scale==0):
            continue
        ops.append(['MULTIPLY&ADD',scale,I,i])
        tmpI=SCALE(A,I,scale)#getting scaled value in tmpI.
        tmpAdd=[round(x+y,prcx) for x,y in zip(tmpI,A[i])]#element-wise addition. Thanks Python!
        A[i]=tmpAdd#update the ith row.
    return A,ops
def CreateX(A,free,pivot,n,m):
    equations=[]
    revpivot=pivot[::-1]
    X=[1 for i in range(0,m)]
    for i in free:
        if(i<m):
            X[i]='X['+str(i)+']'
    for i,k in revpivot:
        equation='X['+str(k)+']'+'='+str(A[i][-1])
        for j in range(k+1,m):
            if( A[i][j]==0):
                continue
            equation+='-'+str(A[i][j])+'*'+str(X[j])
        equations.append(equation)
    return equations
def isInconsistent(basic,m):
    for i in basic:
        if(i==m):
            return True
    return False

