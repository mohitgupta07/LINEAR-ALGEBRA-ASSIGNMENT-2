import sys
out=[sys.stdout]
def printer(*args,**kwargs):
    for o in out:
        print(*args,**kwargs,file=o)
def appendToOut(fp):
    print("appending to list:",fp)
    out.append(fp)
