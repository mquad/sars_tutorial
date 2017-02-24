import treelib
from treelib.tree import  NodeIDAbsentError
import uuid

class SmartTree(treelib.Tree):

    _PATH_NOT_FOUND = -1

    def find_path(self,origin, path):
        """Takes the nodeId where to start the path search and the path to look for,
        :returns -1 if path not found, nodeId of the last node if path found"""

        if not path:
            #path found
            return origin

        res = self._PATH_NOT_FOUND

        for nodeId in self[origin].fpointer:
            node = self[nodeId]
            if (node.tag == path[0]):
                res = self.find_path(nodeId, path[1:])
                break

        if res == None:
            #path not found
            return self._PATH_NOT_FOUND
        else:
            return res


    def longest_subpath(self,origin,path):
        """Takes the nodeId where to start the path search and the path to look for,
        :returns the nodeId of the node where the path is broken and the number of missing element for the complete path"""

        if not path: #path empty, all nodes matched
            #path found
            return (origin,0)

        res = ()

        for nodeId in self[origin].fpointer: #se ci sono molti nodi figli qui diventa lento
            node = self[nodeId]
            if (node.tag == path[0]):
                res = self.longest_subpath(nodeId, path[1:])
                break

        if res == ():
            #path not found
            return (origin,len(path))
        else:
            return res


    def add_path(self,origin,path,support=None):
        """add a path, starting from origin"""
        sub = self.longest_subpath(origin,path)
        if(sub[1] == 0):
            #path aready exists, updating support
            self[sub[0]].data = {'support': support}

        else:
            # add what's missing
            missingPath = path[-sub[1]:]

            par = sub[0]
            for item in missingPath:
                itemId = uuid.uuid4()
                self.create_node(item,itemId,parent=par,data={'support':support})
                par = itemId

    def path_is_valid(self,path):
        return path != self._PATH_NOT_FOUND

    def create_node(self, tag=None, identifier=None, parent=None, data=None):
        """override to get a random id if none provided"""
        id = uuid.uuid4() if identifier == None else identifier
        if id == self._PATH_NOT_FOUND:
            raise NameError("Cannot create a node with special id "+str(self._PATH_NOT_FOUND))
        super(SmartTree, self).create_node(tag,id,parent,data)


    def set_root(self,rootTag=None,rootId=None):
        id = uuid.uuid4()
        rootId = rootId if rootId != None else id
        rootTag = rootTag if rootTag != None else 'root'
        self.create_node(rootTag,rootId)
        self.root = rootId
        return rootId

    def get_root(self):
        try:
            return self.root
        except AttributeError:
            return None

    def find_n_legth_paths(self,origin,length,excludeOrigin = True):

        if length == 0:
            return [[]] if excludeOrigin else [[origin]]

        else:
            children = self[origin].fpointer
            paths = []
            for c in children:
                childrenPaths = self.find_n_legth_paths(c,length-1,False)
                # this line is magic, if there are no children the all path gets lost, that's how i get paths of exactly length wanted
                l = list(map(lambda x:[]+x,childrenPaths)) if excludeOrigin else list(map(lambda x:[origin]+x,childrenPaths))
                for el in l:
                    paths.append(el)
            return paths

    def get_paths_tag(self,listOfPaths):
        return list(map(lambda x: self.get_nodes_tag(x),listOfPaths))

    def get_nodes_tag(self,listOfNids):
        return list(map(lambda y: self[y].tag,listOfNids))