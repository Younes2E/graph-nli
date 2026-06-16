from text_span_v2 import construction, mwu, relation, document, getobj, name
from igraph import *

GRAPH_DEP_SUFX= '_igraph.dot'
#DEBUG = True
DEBUG = False

def make_graph( label_attr = 'lbl', root_label = '[GRAPH_ROOT]' ):
    g = Graph( directed = True )
    g.vs[ label_attr ] = []
    g.es[ label_attr ] = []
    root_v = g.add_vertex( lbl = root_label )
    return g

def add_labelled_vertice(  gr, l, last_v = False ):
    # gr is an igraph graph
    # l is the label (str) of the new vertex
    # s is the "start_attribute"
    global label_attr
    global root_label
    global dir_attr
    # since word may label nodes, there may be two nodes with the same label in the graph (see below)
    #if  not gr.vs.select( lbl=l ):
        # alternate a leading newline in the labels to avoid graphic overlap
    if last_v:
        l = '\n\n' + l
        last_v = False
    new_vertice = gr.add_vertex( lbl = l )
    new_v = len( gr.vs ) - 1
    assert( gr.vs[ new_v ].index == new_v )
## only used when all nodes have distinc labels
##    assert( new_vertice == gr.vs.select( lbl=l )[0] )
    assert( new_vertice in gr.vs.select( lbl=l ) )
    if DEBUG:
        print( 'add_labelled_vertice() node id {0} label {1}'.format( new_vertice.index, new_vertice['lbl'] ))
    return new_vertice
## only used when no two nodes have the same label (see above)
##    else:
##        existing_vertices = gr.vs.select( lbl=l )
##        assert( len( existing_vertices ) < 2 )
##        return existing_vertices[ 0 ]

def is_root( g, v ):
    # if (g.es.select( _source = root_index, _from = v.index)) or (g.es.select( _source = v.index, _from = root_index )):
    return v.index == 0
    
def color_idx( g, v, label_attr ):
    #print( 'DEBUG color_idx====>' + '{0}'.format(  v[ label_attr ] ))
    if is_root( g, v):
        return 2
    if v[ label_attr ] == None:
        return  1
    elif v[ label_attr ].split( '|' )[ 1 ].strip() in [ 'NOUN', 'PROPN' ]: 
        return 5
    elif v[ label_attr ].split( '|' )[ 1 ].strip()   == 'VERB': 
        return 2
    elif v[ label_attr ].split( '|' )[ 1 ].strip()   == 'NounGroup':
        return 3
    else:
        return 4

def edge_color_idx( g, e, label_attr ):
    print( 'DEBUG edge_color_idx ====>' + '{0}'.format(  e[ label_attr ] ))
    if e[ label_attr ] == None:
        return 0
    if e[ label_attr ].split( '_' )[ 0 ] == 'NextW':
        return 6
    elif e[ label_attr ] == 'PartOfNounGroup':
        return 3
    else:
        return 0

def show( g, layout = None, bbox = None, v_size = None, lbl_size = None, margin = None, label_attr = 'lbl' ):
    visual_style = {}
    if v_size:
        visual_style["vertex_size"] = v_size
    ##g.vs["color"] = [color_dict[gender] for gender in g.vs["gender"]]
    color_dict = { 0:'black', 1:'white', 2:'red', 3:'green', 4:'cyan', 5:'orange', 6:'grey', 7:'purple', 8:'brown', 9:'Daffodil', 10:'yellow', 11:'magenta'}
    visual_style["vertex_color"] = [ color_dict[ k ] for k in list( map( lambda v : color_idx( g, v, label_attr), g.vs )) ]
    visual_style["vertex_label"] = g.vs[ str('lbl') ]
    #visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    #visual_style["shape"] = 'rectangle'
    visual_style[ "vertex_label_dist" ] = 1
    if lbl_size:
        visual_style["label_size"] = lbl_size
    if layout:
        visual_style["layout"] = layout
    if bbox:
        visual_style["bbox"] = bbox
    if margin:
        visual_style["margin"] = margin
    visual_style["edge_color"] = [ color_dict[ k ] for k in list( map( lambda e : edge_color_idx( g, e, label_attr), g.es )) ]
    plot(g, **visual_style)

def make_word_node_label( m, doc ):
    assert( type( m ) is mwu )
    return m.text( doc ) + ' | ' + '_'.join( m.annotations[ 'pos' ] )

def root_relocated_mwu_nm( m_nm, root_nm ):
    # a name is a pair ( sid, tokid), and all the sid have the same value.
    sid = root_nm[ 0 ]
    nm_sid_0 = (sid, 0)
    if m_nm == nm_sid_0:
        return root_nm
    elif m_nm  == root_nm:    
        return nm_sid_0
    else:
        return m_nm

def debug_print_vertices( g ):
    print( '\nDEBUG vertices g.vs== {0}'.format( g.vs ))
    for v in g.vs:
        print( 'vertex {0}'.format(   v ))
    print( '__________' )

def debug_print_edges( g ):
    print( '\n DEBUG edges g.es== {0}'.format( g.es ))
    for e in g.es:
        print( 'edge {0}'.format( e ))
    print( '__________' )
    
def spacy_display_sent_graph( snt, doc, show_graph = False, target_file = '/tmp/igraph_out.pdf'):
    assert( type( snt ) is construction )
    assert( type( doc ) is document )

    if DEBUG:
        print( 'DEBUG  in spacy_display_sent_graph name(snt)== {0}'.format( name( snt )  ))
    # left first breadth first exploration of the dependency tree
    root_rel = [ r  for r in snt.rels.values() if r.typ == 'ROOT' ][ 0 ]
    
    #curr_level_mwu_nms = [ name( m ) for m in snt.mwus.values() if m.annotations[ 'deprel'] == 'ROOT' ]
    # NOTE: add  explicitely the sentence id to make a key pair because varify() and getodj require unique ids for all instances inside a class.
    curr_level_mwus = [ r.trgt for r in snt.rels.values() if r.typ == 'ROOT' ]

    if DEBUG:
        print( 'DEBUG pap last curr_level_mwus {0}'.format( curr_level_mwus ))
        
        for m in snt.mwus.keys():
            print('\t\tDEBUG a mwu from m== {0} m.annotations == {1}'.format( m, getobj(m).annotations ))
        for r in snt.rels.keys():
            print('\t\tDEBUG a rel from r== {0} r.typ == {1}'.format( r, getobj( r ).typ ))
        print( 'DEBUG spacy_display_sent_graph at ROOT node curr_level_mwu== {0}'.format( curr_level_mwus ))
        
    assert( len(curr_level_mwus) == 1 )

    if DEBUG:
        print( 'DEBUG  type(root_rel.trgt)==   {0} root_rel.trgt== {1}'.format( type( root_rel.trgt ), root_rel.trgt ))
        
    root_mwu = root_rel.trgt
    if DEBUG:
        print( 'root_mwu_nm  {0}'.format( name( root_mwu )))
    
    mwu_labels = []
    rel_labels = []
    # node attributes
    graph_root_index = 0
    graph_root_label = make_word_node_label( root_mwu, doc )
    g = make_graph( label_attr = 'label' )
    graph_root_vertex = g.vs[ 0 ]  # root of the dep tree must have 0 index to be displayed on top of the plot
    mwu_vertex_map = { name( root_mwu ) : graph_root_vertex }
    assert( graph_root_vertex[ 'label' ]  == None )
    assert( graph_root_label not in mwu_labels )
    graph_root_vertex[ 'label' ]  = graph_root_label
    mwu_labels.append( graph_root_label  )
    g.add_edge( source = graph_root_vertex,
                target = graph_root_vertex )
    assert( g.es[ -1 ][ "label" ] == None )
    root_edge_label  = name( root_rel ) + root_rel.typ
    assert( root_edge_label not in rel_labels )
    g.es[ -1 ][ "label" ] = root_edge_label
    rel_labels.append( name( root_rel ) + root_rel.typ )
    processed_mwu_nms = [ ]
    processed_rel_nms = [ name( root_rel ) ]
    #-------- graph drawing and display -------------

    # a list of  pairs: (sid, mwu_id )
    new_level_mwus = [ root_mwu ]
    
    while len( curr_level_mwus ) > 0:
        if DEBUG:
            print( '\n\n___________________________________________________\n\n' )
            print( 'uDEBUG curr_level_mwus has len {0} with content {1}'.format( len( curr_level_mwus ), curr_level_mwus))
            print( 'DEBUG vertex_map {0}'.format( mwu_vertex_map ))
        # --------step 1: sort the level mwus from left to right and create the graph vertices
        for m in sorted( curr_level_mwus, key = lambda x : (x.txtspans[0][0], x.txtspans[-1][1]) ):
            if (m != root_mwu) and (name( m ) not in processed_mwu_nms):
                graph_node_label = make_word_node_label( m, doc )
                g.add_vertex( lbl = graph_node_label )
                v = g.vs[ -1 ]
                assert(  v[ "label" ] == None  )
                assert( graph_node_label not in mwu_labels )
                v[ "label" ] =  graph_node_label
                mwu_labels.append( graph_node_label  )
                mwu_vertex_map[ name( m ) ] = v
                processed_mwu_nms.append( name( m ) )
        if DEBUG:
            debug_print_vertices( g )
        #--------step 2: create the edges for the current level
        for r in [ a_r for a_r in snt.rels.values() ]:
            if (r.src in curr_level_mwus) and (r.name not in processed_rel_nms):
                if mwu_vertex_map[ name( r.src ) ] is None:
                    print( '>>>>>>>>>> DEBUG BAD src nm {0}'.format( name( r.src ) ))
                assert( mwu_vertex_map[ name( r.src )  ] is not None )
                if name( r.trgt ) not in new_level_mwus:
                    new_level_mwus.append( r.trgt )
                #--------- create the edge target vertex
                if name( r.trgt ) not in mwu_vertex_map.keys():
                    graph_node_label = make_word_node_label( r.trgt, doc )
                    g.add_vertex( lbl = graph_node_label )
                    v = g.vs[ -1 ]
                    assert(  v[ "label" ] == None  )
                    assert( graph_node_label  not in mwu_labels )
                    v[ "label" ] =  graph_node_label 
                    mwu_labels.append( graph_node_label  )
                    mwu_vertex_map[ name( r.trgt ) ] = v
                    processed_mwu_nms.append(  name( r.trgt ) )
                else:
                    pass
                g.add_edge( source = mwu_vertex_map[ name( r.src  ) ],
                            target = mwu_vertex_map[ name( r.trgt ) ],
                            lbl = rel_labels[ -1 ] )
                e = g.es[ -1 ]
                assert(  e[ "label" ] == None  )
                graph_edge_label = name( r ) + r.typ
                assert( graph_edge_label not in rel_labels )
                e[ "label" ] =  graph_edge_label
                rel_labels.append( name( r ) + r.typ )
                processed_rel_nms.append(  name( r ) )
            else:
                pass
        #-------- update 3: the list of mwus of the next level
        if DEBUG:
            print( 'SHOULD be empty after a while DEBUG new_level_mwu_nms has len {0} with content {1}'.format( len( new_level_mwus ),
                                                                                                                new_level_mwus))
        curr_level_mwus = new_level_mwus
        
        if DEBUG:
            print( 'NEW  curr_level_mwu_nms has len {0} with content {1}'.format( len( curr_level_mwus ), curr_level_mwus))
        new_level_mwus = []

        if DEBUG:
            debug_print_edges( g )

    # using vertex_label= is equivalent to g.vs=...,  so to label your edges, use g.es=:
    # g.es[ "label" ] = ["A", "B", "C"] or g.es["name"] = map(str, np.arange(N))
##    g.vs[ "label" ] = mwu_labels
##    g.es[ "label" ] = rel_labels

    #lay= g.layout( 'rt', root=[0] )
    #lay = g.layout( 'fr' )  
    #lay = g.layout( 'tree', root=[0]  ) 
    #lay = g.layout( 'rt_circular', root=[0] ) 
    lay = g.layout( 'kk' )
    autocurve( g )  # prevent edges from overlapping
    #lay.rotate( 90 ) # horizontal versus vertical display
    
    if show_graph:
        show( g, layout=lay, bbox= (4096, 4096), margin=150, v_size = 10 )

    target_file  = name( doc ) + '_' + name( snt ) + GRAPH_DEP_SUFX

    if DEBUG:
        debug_print_vertices( g )

    # view with xdot application
    dot = g.write( target_file )


 #------------------------- end of graph drawaing defs----
