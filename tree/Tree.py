import treelib
from treelib.tree import  NodeIDAbsentError
import uuid

class SmartTree(treelib.Tree):

    (ROOT, DEPTH, WIDTH, ZIGZAG) = list(range(4))


    def __real_true(self, p):
        return True

    def expand_tree(self, nid=None, mode=DEPTH, filter=None, key=None,
                    reverse=False):
        """
        Python generator. Loosly based on an algorithm from
        'Essential LISP' by John R. Anderson, Albert T. Corbett, and
        Brian J. Reiser, page 239-241
        UPDATE: the @filter function is performed on Node object during
        traversing. In this manner, the traversing will not continue to
        following children of node whose condition does not pass the filter.
        UPDATE: the @key and @reverse are present to sort nodes at each
        level.
        """
        nid = self.root if (nid is None) else nid
        if not self.contains(nid):
            raise NodeIDAbsentError("Node '%s' is not in the tree" % nid)

        filter = self.__real_true if (filter is None) else filter
        if filter(self[nid]):
            yield nid
            queue = [self[i] for i in self[nid].fpointer if filter(self[i])]
            if mode in [self.DEPTH, self.WIDTH]:
                queue.sort(key=key, reverse=reverse)
                while queue:
                    yield queue[0] #return the node
                    expansion = [self[i] for i in queue[0].fpointer
                                 if filter(self[i])]
                    expansion.sort(key=key, reverse=reverse)
                    if mode is self.DEPTH:
                        queue = expansion + queue[1:]  # depth-first
                    elif mode is self.WIDTH:
                        queue = queue[1:] + expansion  # width-first

            elif mode is self.ZIGZAG:
                # Suggested by Ilya Kuprik (ilya-spy@ynadex.ru).
                stack_fw = []
                queue.reverse()
                stack = stack_bw = queue
                direction = False
                while stack:
                    expansion = [self[i] for i in stack[0].fpointer
                                 if filter(self[i])]
                    yield stack.pop(0)
                    if direction:
                        expansion.reverse()
                        stack_bw = expansion + stack_bw
                    else:
                        stack_fw = expansion + stack_fw
                    if not stack:
                        direction = not direction
                        stack = stack_fw if direction else stack_bw

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

        if path == []:
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


    def add_path(self,path,support,origin):
        """add a path, starting from origin"""
        sub = self.longest_subpath(origin,path)
        if(sub[1] == 0):
            #path aready exists, updating support
            self[sub[0]].data = {'support':support}

        else:
            # add what's missing
            missingPath = path[-sub[1]:]

            par = sub[0]
            for item in missingPath:
                itemId = uuid.uuid4()
                self.create_node(item,itemId,parent=par)
                par = itemId


    def create_node(self, tag=None, identifier=None, parent=None, data=None):
        id = uuid.uuid4() if identifier == None else identifier
        super(SmartTree, self).create_node(tag,id,parent,data)