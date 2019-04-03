
def solver(A,n,m):#A is square Matrix
    #remember n==m in this case so beware of conditions...
    #Step1:- Reduce to Upper Triangler matrix <echolen form> Also memorizing each operation
    Operations=[]
    K=-1
    I=-1
    free=[]
    basic=[]
    pivot=[]
    #print(A)
    NotignoreThis=False
    while(K<m and I<n):
        I+=1#row for pivot...
        K+=1#Column at which pivot will occur
        if(I==n or K==m):#remove this and make K<(m-1) and I<(n-1)
            break
        flag=0
        while(A[I][K]==0):
            #print(I,K)
            flag=0
            for i in range(I+1,n):                
                if(A[i][K]!=0):#checking for non-zero column for pivot possibility
                    Operations.append(['SWITCH',I+1,i+1])
                    flag=1
                    #perform switch
                    tmp=A[I]
                    A[I]=A[i]
                    A[i]=tmp
                    break
                pass#end of for loop...
            #add free variable and move on.
            if(flag==0):
                free.append(K)
                K+=1
            if(K>=m or I>=n):#Remember square matrix so K>=m unlike K>m in first Problem
                #Ops... No more pivot left.
                flag=2#since I dont know how to add label so better use flag to get out.
                break
            #print(I,K, "n",n,m)
        pass#end of while loop.
    
        if(flag==2):
            #CRAP! need to exit the loop...
            break
        #let's roll!
        #We now have a non-zero column entry so add a basic variable to our list
        #print(I,K,flag)
        basic.append(K)
        pivot.append((I,K))
        #Converting the row below the pivot K to 0.
        scale=1/float(A[I][K])
        #scale*=scale
        if scale!=1:
            Operations.append(['MULTIPLY',scale,I+1])#adding to operations list.
        A[I]=SCALE(A,I,scale)#scaling done...returns a row
        #print(A)
        A,ops,flag=ConvertRowsToZero(A,I,K,n,NotignoreThis)
        if(len(ops)!=0):#sometimes zero operations can occur
            for op in ops:
                Operations.append(op)
        if(NotignoreThis and flag==1):
            #break as it's not inverse matrix
            break
    pass#end of outer while loop
    #Add those free variables which were unreachable in step1 and couldnt get added.
    for i in range(basic[-1]+1,m):
        free.append(i)# add i to free list...
        
    #Step2:-Convert into identity matrix if possible..simply reduced echolen form...
    #to do so make each column zero which has a basic variable(pivot) in it and obviously except the pivot.
    for i,k in pivot:
        A,ops=MakeItReduced(A,i,k)
        if(len(ops)!=0):#sometimes zero operations can occur
            for op in ops:
                Operations.append(op)        
    #step3:-Convert identity matrix into Ainv...
    #But first let's check if the given A is converted  to identity in reduced echolen form.
    #isIdentity(A) <- No need for this as we can just check for free variable.
    #so IF no. of free variable ==0 then identity matrix is guranteed in case of square matrix
    #f=open('output_problem2.txt','w')
    if(len(free)==0):
        #identityMatrix...
        AInv=getInv(n,Operations)
        #AInv=A
        #print('YAAY! FOUND ONE!')
        #f.write('YAAY! FOUND ONE!\n')
        for i in AInv:
            sx=' '.join(map(str,i))
            #f.writelines([sx,'\n'])
            #print(sx)
    else:
        #print('ALAS! DIDN’T FIND ONE!')
        #f.write('ALAS! DIDN’T FIND ONE!\n')
        pass
    #tmp='\n'.join(map(str,[' '.join(map(str,i)) for i in Operations]))
    #print(tmp)
    #f.write(tmp)
    for i in Operations:
        sx=' '.join(map(str,i))
        #f.writelines([sx,'\n'])
        #print(sx)
        if  len(free)==0 and i[0]!='SWITCH':
            #f.writelines([sx,'\n'])
            #print(sx)
            pass
    #f.close() 
    #print A
    #end of COdE solver
    return AInv


def SCALE(A,I,scale):
    row=[i*scale for i in A[I]]
    return row

def ConvertRowsToZero(A,I,K,n,NotignoreThis):
    global ADD
    ops=[]
    flag=0
    #print(A,"float??",A[0][1])
    for i in range(I+1,n):
        #print(A[i])
        scale=-1*float(A[i][K])#**2
        if(scale==0):
            continue
        ops.append(['MULTIPLY&ADD',scale,I+1,i+1])
        tmpI=SCALE(A,I,scale)#getting scaled value in tmpI.

        tmpAdd=[x+y for x,y in zip(tmpI,A[i])]#element-wise addition. Thanks Python!
        A[i]=tmpAdd#update the ith row.
        count=0
        for i in tmpAdd:
            if(i==0):
                count+=1
        if(NotignoreThis and count==n):
            flag=1
            break
    #print(ops)    
    return A,ops,flag
        
def MakeItReduced(A,I,K):
    global ADD
    ops=[]
    for i in range(0,I):
        #print(A[i])
        scale=-1*float(A[i][K])#**2
        if(scale==0):
            continue
        ops.append(['MULTIPLY&ADD',scale,I+1,i+1])
        tmpI=SCALE(A,I,scale)#getting scaled value in tmpI.

        tmpAdd=[x+y for x,y in zip(tmpI,A[i])]#element-wise addition. Thanks Python!
        A[i]=tmpAdd#update the ith row.
    #print(ops)    
    return A,ops



def isInconsistent(basic,m):
    for i in basic:
        if(i==m):
            return True
    return False
    
def isIdentity(A):
    n=len(A)
    for i in range(0,n):
        if(A[i][i]!=1):
            return False
    return True

def getInv(n,Operations):
    
    A=[]
    for i in range(n):
        tmp=[0 for j in range(n)]
        tmp[i]=1
        A.append(tmp)
    #print(Ainv)Identity generated...
    for op in Operations:
        if(op[0]=='SWITCH'):
            #switch...
            I=op[1]-1
            i=op[2]-1
            tmp=A[I]
            A[I]=A[i]
            A[i]=tmp
        elif(op[0]=='MULTIPLY'):
            scale=float(op[1])
            I=op[2]-1
            row=SCALE(A,I,scale)
            A[I]=row
        elif(op[0]=='MULTIPLY&ADD'):
            scale=float(op[1])
            I=op[2]-1
            i=op[3]-1
            tmpI=SCALE(A,I,scale)#getting scaled value in tmpI.
            tmpAdd=[x+y for x,y in zip(tmpI,A[i])]#element-wise addition. Thanks Python!
            A[i]=tmpAdd#update the ith row
        A=[[round(i,3) for i in tmp ] for tmp in A]
    return A #AInverse


'''0.1 0.3 0.2 0.4 27
0.3 0.3 0.2 0.1 21
0.4 0.1 0.4 0.1 22
0.1 0.2 0.1 0.2 16'''

'''
0.1 0.3 0.2 0.4 
0.3 0.3 0.2 0.1 
0.4 0.1 0.4 0.1 
0.1 0.2 0.1 0.2
'''

'''
27 21 22 16
0.1 0.3 0.2 0.4
0.3 0.3 0.2 0.1
0.4 0.1 0.4 0.1
0.1 0.2 0.1 0.2
33 35 46 57
'''

'''
3 2
10 20 30
0.1 0.2
0.3 0.4
0.2 0.2
100 100
'''
