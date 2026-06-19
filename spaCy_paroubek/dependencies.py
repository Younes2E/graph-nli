import re

from text_span_v2 import mwu, relation, construction, document, name, getobj

from anytree import Node, NodeMixin, RenderTree,  PreOrderIter

DEPNODEPRFX = 'depNode'
ANNOT_DEPTREE_KEY = 'anytree_node'

SENTPRFX = 'S'

#DEMO = True
DEMO = False

#DEBUG = True
DEBUG = False

# head is synonymous of src and dependent is synonymous of trgt
NODE_ROLES = [ 'src', 'trgt', 'root', 'any', 'head', 'dependent' ]

def make_word_dep_node_label( m, doc ):
    assert( type( m ) is mwu )
    # return m.text( doc ) + ' | ' + '_'.join( m.annotations[ 'pos' ] )
    return name( m )

UNDEF = 'UNDEFINED'
class dep_node( construction, NodeMixin ):
    def __init__( self, name = UNDEF, length= 1, width= 1, parent=None, children=None, mwus = {}, rels = {}, kstructs = {}):
        self.name = name
        self.length = length
        self.width = width
        self.parent = parent
        self.mwus = {}
        self.rels = {}
        if children:
            self.children = children
        super(dep_node, self).__init__( nm = name, mwus = mwus, rels= rels )
        construction.__init__( construction(self), nm = name, mwus = mwus, rels= rels )
        NodeMixin.__init__( self )
        
        if len( mwus ) == 0:
            pass
        elif len( mwus) == 1:
            if name ==  UNDEF:
                self.name = name( mwus[ 0 ] )
            else:
                try:
                    if rels:
                        root_mwu = [ r for r in rels if r.typ == 'ROOT' ][ 0 ].src
                        self.name = name( root_mwu )
                    else:
                        root_mwu = None
                except ValueError as error:
                    print( 'ERROR no root ROOT dependence in dependencies: {0}'.format( rels ))
                    return None
        assert( len( self.mwus.keys()) < 2 )
        
    def __repr__( self ):
           msg = ''
##           for pre, fill, node in RenderTree( self.root ):
##               tree_str = u"%s%s" % (pre, name( node ))
##               msg += tree_str
           msg = self.name
           return msg

    def __str__( self ):
           return self.__repr__()

    def add_dep( self, r ):
        assert( (type( dep) is relation) and (dep.type == 'deprel' ))
        if  self.root.name == 'UNDEF':
            assert( (self.typ == '') and (len( self.mwus) == 0) and (len( self.rels) == 0) )
            assert( len( self.rels ) == 0 )
            self.rels  = { name( r ) : r }
            self.root.name = make_word_dep_node_label( r.src )
            self.mwus.append( r.src )
            new_dependent = make_word_dep_node_label( r.trgt )
            new_dependent.mwus.append( r.trgt )
            new_dependent.parent = self
            self.children.append( new_dependent )
        else:
            self.rels[ name( r ) ] =  r
            if name( r.src ) not in self.mwus:
                self.mwus.append( r.src )
        

    def remove_dep( self, dep_nm = '' ):
        assert( type( dep_nm ) is str )
        del self.rels[ dep_nm ]

    def find_node_mwu( self, dep_node = None, pred = lambda x,y : False ):
        if dep_node == None:
            dep_node = self.root
        else:
            pass
        for node in PreOrderIter( dep_node ):
            if pred( node ):
                 return node
        return None

    def find_dep( self, dep_node = None, role = '', pred = lambda x : False ):
        if dep_node == None:
            dep_node = self.root
        else:
            pass
        for node in PreOrderIter( dep_node ):
            for r in node.rels:
                if role == 'any':
                    return pred( r )
                elif role == 'root':
                    if r.typ == 'ROOT':
                        assert( r.src == r.trgt )
                        return pred( r.src )
                    else:
                        pass
                elif role in [ 'src', 'head' ]:
                     if pred( r.src ):
                        return r
                     else:
                        pass
                elif role in [ 'trgt', 'dependent' ]:
                     if pred( r.trgt ):
                        return r
                     else:
                        pass
                else:
                    assert( FALSE )
        return None
       
    def text_find_node( self, regex = '', mode = 'search',  dep_node =  None ):
        assert( type( regex ) == str )
        
        def node_match( mwu ):
            try:
                dfsa = re.compile( regex )
                if mode == 'search':
                    return dfsa.search ( mwu.text( self ) )
                else:
                    return dfsa.match( mwu.text( self ) )
            except Exception as error:
                print('caught this exception: ' + repr(error) )
                return None

        if dep_node == None:
            dep_node = self.root
        else:
            pass
        for node in PreOrderIter( dep_node ):
            if node_match( node ):
                 return node
            else:
                pass
                
#---end of class dep_node

def make_sent_dep_tree( snt, doc ):
    # building the sentence dependencies tree with anytree module
    assert( type( snt ) is construction )
    assert( type( doc ) is document )

    # left first breadth first exploration of the dependency tree
    root_mwu = [ m for m in snt.mwus.values() if m.annotations[ 'deprel' ] == 'ROOT' ][ 0 ]
    # variables used for the graph view
    mwu_labels = []
    rel_labels = []
    # node attributes
    graph_root_index = 0
    graph_root_label = make_word_dep_node_label( root_mwu, doc )
    mwu_labels.append( graph_root_label  )
    root_rel = [ r  for r in snt.rels.values() if r.typ == 'ROOT' ][ 0 ]
    rel_labels.append( root_rel.typ )
    processed_rel_nms = [ name( root_rel ) ]
    #-------- graph drawing and display -------------

    # initialize the graphe view root node   
    assert( root_mwu == root_rel.src )
    
    root_mwu.annotations[ ANNOT_DEPTREE_KEY ] = dep_node( make_word_dep_node_label( root_mwu, doc ), parent = None, mwus = { name( root_mwu ) :root_mwu } )
    curr_level_mwus = [ root_mwu ]
    new_level_mwus = [ ]

    # pap 20260421
    DEBUG = True
    # create all the graph view vertices for all relations
    while len( curr_level_mwus ) > 0:
        if DEBUG:
            print('\nDEBUG  make_sent_dep_tree;  beg. while curr_level_mwu_nms == {0}\n'.format(  curr_level_mwus  ))
        # --------step 1: sort the level mwus from left to right and create the graph vertices
        sorted_mwus = sorted( curr_level_mwus,
                               key = lambda x : (x.txtspans[0][0], x.txtspans[-1][1]) )
        new_level_mwus = []
        #--------step 2: create the edges for the current level
        for r in snt.rels.values():
            print( 'name(r )=={0} name( r ) in processed_rel_nms== {1}'.format(name( r ),
                                                                               (name( r ) in processed_rel_nms)))
            if name( r ) in processed_rel_nms:
                pass
            else:
                if DEBUG:
                    print( 'DEBUG make_sent_dep_tree src dep {0}'.format(  name( r.src ) ))
                if  r.src in curr_level_mwus:
                    new_level_mwus.append( r.trgt )
                    if r != root_rel:
                        if DEBUG:
                            print( '\t\tDEBUG make_sent_dep_tree; ADDING edge dep src {0} trgt {1}'.format(  name( r.src ), name( r.trgt )))

                        r.trgt.annotations[ ANNOT_DEPTREE_KEY ] = dep_node( make_word_dep_node_label( r.trgt, doc ),
                                                                         parent = r.src.annotations [ ANNOT_DEPTREE_KEY ],
                                                                         mwus  = { name( r.trgt) : r.trgt } )                           
                        rel_labels.append( r.typ )
                    else:
                        pass
                    processed_rel_nms.append( name( r ) )
                    if  r.trgt not in new_level_mwus:
                        new_level_mwus.append( r.trgt )
                    else:
                        pass
                else:
                    pass
        curr_level_mwus = new_level_mwus
        #-------- update 3: the list of mwus of the next level
        
    all_rels = [ getobj( r_nm ) for r_nm in snt.rels ]
    # pap DEBUG
    for r in all_rels:
        if name( r ) not in processed_rel_nms:
            print( '>>>>>not processed rel ', r )
    assert( False not in [ (name( r ) in processed_rel_nms) for r in all_rels ] )
    if DEBUG:
        for pre, fill, node in RenderTree( root_mwu.annotations[ ANNOT_DEPTREE_KEY ] ):
          treestr = u"%s%s" % (pre, node.name)
          print( treestr )
    # pap 20260421
    DEBUG = False
    return root_mwu.annotations[ ANNOT_DEPTREE_KEY ]

## A Gorn address (Gorn, 1967) is a method of identifying and addressing any node within a tree data structure.
## This notation is often used for identifying nodes in a parse tree defined by phrase structure rules.
##
## The Gorn address is a sequence of zero or more integers conventionally separated by dots, e.g., 0 or 1.0.1.
## The root which Gorn calls * can be regarded as the empty sequence.
## And the j-th child of the i-th child has an address i . j, counting from 0.
##
## It is named after American computer scientist Saul Gorn.
##
## References:
## Gorn, S. (1967). Explicit definitions and linguistic dominoes. Systems and Computer Science, Eds. J. Hart & S. Takasu. 77–115. University of Toronto Press, Toronto Canada.

def show_node( a_node, doc ):
     print( a_node +  '/' + list(a_node.mwus.values())[ 0 ].text( doc ) )
     return name( a_node )  +  '/' + list(a_node.mwus.values())[ 0 ].text( doc )

def foo( a_node ):
     POSKEY =  'pos'
     the_mwu = (list( a_node.mwus.values()))[ 0 ]
     print('the_mwu.annotations[ POSKEY ]== {0}'.format( the_mwu.annotations[ POSKEY ]  ))
     return (the_mwu.annotations[ POSKEY ] in [ 'NOUN', 'PROPN' ])
    
def x_down_level_recur( curr_node_lst = [], sorted_next_level_node_lst = [], doc = None, nlevel= 0,
                        res = [], level_node_lst_proc_fun = lambda x,y,z: x,
                        ng_root_p = foo ):
    #if DEBUG:
    print( '\nDEBUG x_down_level_recur \n\t\tcurr_node_lst== ', curr_node_lst, '\nnlevel= ', nlevel,'\nres= ', res )
    for z in curr_node_lst:
        print( '\tTESTB a_node_lst[ 0 ]== {0} ng_root_p( a_node_lst[ 0 ] )== {1}'.format( z, foo( z )))
    return x_breadth_level_recur( a_node_lst = sorted_next_level_node_lst,
                                  curr_node_lst= [],
                                  doc = doc,
                                  nlevel = nlevel + 1,
                                  res = res + level_node_lst_proc_fun( curr_node_lst, nlevel, doc ),
                                  level_node_lst_proc_fun = level_node_lst_proc_fun )


def x_breadth_level_recur( a_node_lst = [], curr_node_lst = [], next_level_node_lst = [], doc = None, nlevel = 0,
                           res = [], level_node_lst_proc_fun = lambda x,y,z: x,
                           ng_root_p = foo):
    print('nlevel== {0} len( a_node_lst== {1}'.format( nlevel,  nlevel  ))
##    assert( ((nlevel == 0) and (len( a_node_lst ) == 1)) or
##            ((nlevel > 0)  and (len( a_node_lst ) >= 1)) )
    # building of the next current horizontal level in the dependence tree
    #if DEBUG:
    if nlevel == 0:
        print( 'TEST a_node_lst[ 0 ]== {0} ng_root_p( a_node_lst[ 0 ] )== {1}'.format( a_node_lst[ 0 ], ng_root_p( a_node_lst[ 0 ] )))
        #sorted_next_level_node_lst = sorted( [ x for x in a_node_lst ], key = lambda nd : (list( nd.mwus.values())[ 0 ]).txtspans[0][0] )
        sorted_next_level_node_lst = sorted( [ x for x in a_node_lst if x not in a_node_lst], key = lambda nd : (list( nd.mwus.values())[ 0 ]).txtspans[0][0] )
        if ng_root_p( a_node_lst[ 0 ] ):
            if len( sorted_next_level_node_lst ) > 0:
                return x_down_level_recur( curr_node_lst = [ a_node_lst[ 0 ] ],
                                           sorted_next_level_node_lst = sorted_next_level_node_lst,
                                           doc = doc,
                                           nlevel = nlevel,
                                           res = [ a_node_lst[ 0 ] ],
                                           level_node_lst_proc_fun = level_node_lst_proc_fun,
                                           ng_root_p = ng_root_p )
            else:
                return [ a_node_lst[ 0 ] ]
        else:
            if len( sorted_next_level_node_lst ) > 0:
                return x_down_level_recur( curr_node_lst = [ a_node_lst[ 0 ] ],
                                           sorted_next_level_node_lst = sorted_next_level_node_lst,
                                           doc = doc,
                                           nlevel = nlevel,
                                           res = [ ],
                                           level_node_lst_proc_fun = level_node_lst_proc_fun,
                                           ng_root_p = ng_root_p )
            else:
                return [ ]
    else:     
        print( '\nDEBUG x_breadth_level_recur( \na_node_lst== ', a_node_lst, '\nnlevel = ', nlevel, '\nres= ', res )
        if len( a_node_lst ) > 0:
            if a_node_lst[ 0 ].children: 
                print( 'TEST a_node_lst[ 0 ]== {0} ng_root_p( a_node_lst[ 0 ] )== {1}'.format( a_node_lst[ 0 ], ng_root_p( a_node_lst[ 0 ] )))
                return x_breadth_level_recur( a_node_lst[ 1: ],
                                              curr_node_lst = curr_node_lst + [ a_node_lst[ 0 ] ],
                                              next_level_node_lst = next_level_node_lst + [ nd for nd  in a_node_lst[ 0 ].children ], # tuple to list conversion
                                              doc = doc,
                                              nlevel = nlevel,
                                              res = res,
                                              level_node_lst_proc_fun = level_node_lst_proc_fun,
                                              ng_root_p =  ng_root_p )
            else:
                return x_breadth_level_recur( a_node_lst[ 1: ],
                                              curr_node_lst = curr_node_lst + [ a_node_lst[ 0 ] ],
                                              next_level_node_lst = next_level_node_lst, # tuple to list conversion
                                              doc = doc,
                                              nlevel = nlevel,
                                              res = res,
                                              level_node_lst_proc_fun = level_node_lst_proc_fun,
                                              ng_root_p =  ng_root_p )
               
        else:
            return []


 
