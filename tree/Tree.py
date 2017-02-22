import treelib
from treelib.tree import  NodeIDAbsentError
import uuid

class SmartTree(treelib.Tree):

    def find_path(self,origin, path):
        """Takes the nodeId where to start the path search and the path to look for,
        :returns -1 if path not found, nodeId of the last node if path found"""

        if path == []:
            #path found
            return origin

        res = -1

        for nodeId in self[origin].fpointer:
            node = self[nodeId]
            if (node.tag == path[0]):
                res = self.find_path(nodeId, path[1:])
                break

        if res == None:
            #path not found
            return -1
        else:
            return res


    def longest_subpath(self,origin,path):
        """Takes the nodeId where to start the path search and the path to look for,
        :returns the nodeId of the node where the path is broken and the number of missing element for the complete path"""

        if not path: #path empty, all nodes matched
            #path found
            return (origin,0)

        res = ()

        for nodeId in self[origin].fpointer:
            node = self[nodeId]
            if (node.tag == path[0]):
                res = self.longest_subpath(nodeId, path[1:])
                break

        if res == ():
            #path not found
            return (origin,len(path))
        else:
            return res


    def add_path(self,origin,path,support):
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



    def create_node(self, tag=None, identifier=None, parent=None, data=None):
        """override to get a random id if none provided"""
        id = uuid.uuid4() if identifier == None else identifier
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
