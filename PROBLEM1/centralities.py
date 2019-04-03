class centralities(object):
    def __init__(self,graph):
        self.adj_list = {}
        self.vertices= list(graph.nodes)
        self.edges=list(graph.edges)
        for u,v in self.edges:
            if u in self.adj_list:
                self.adj_list[u].append(v)
            else:
                self.adj_list[u] = [ v ]
            if v in self.adj_list:
                self.adj_list[v].append(u)
            else:
                self.adj_list[v] = [ u ]
        self.vertex_freq = {}
        for v in self.vertices:
            self.vertex_freq[v] = 0
        self.edge_freq = {}
        for e in self.edges:
            self.edge_freq[e] = 0

        self.visited={x:False for x in self.vertices}

    def BFS(self,u):
        self.visited[u]=True
        tmp_ver=1
        adj_list=[]
        for v in self.adj_list[u]:
            if self.visited[v] == False:
                self.visited[v]=True
                adj_list.append(v)
        
        for v in adj_list:
            a=self.BFS(v)
            tmp_ver+=a
            if (u,v) in self.edge_freq:
                self.edge_freq[(u,v)]+=a
            else:
                self.edge_freq[(v,u)]+=a
        self.vertex_freq[u]+=tmp_ver
        return tmp_ver
    def getCentralities(self):
        for u in self.vertices:
            self.visited={x:False for x in self.vertices}
            self.BFS(u)
        
    
    
