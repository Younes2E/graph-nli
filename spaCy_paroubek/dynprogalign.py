## This code written in 2020 by Patrick Paroubek <pap@limsi.fr> is licensed under CC BY-NC-SA
## This license allows reusers to distribute, remix, adapt, and build upon the material in any medium
## or format for noncommercial purposes only, and only so long as attribution is given to the creator.
## If you remix, adapt, or build upon the material, you must license the modified material under
## identical terms. 
## CC BY-NC-SA includes the following elements:
## BY – Credit must be given to the creator
## NC – Only noncommercial uses of the work are permitted
## SA – Adaptations must be shared under the same terms
## For further information see https://creativecommons.org/about/cclicenses/
## or look into the directory: ./copyright_NOTICES

import io
import sys

from pprint import pprint

from math import inf as maxint

from statistics import median

# this definition applies to characters, tokens, terms, words, multiwords etc.
# for now only 2 levels are needed one with meta_idx==None (i.e. the token/word level)
# and another with meta_idx==index of the token/word level and idx==index of the character level

class indexed_object( object ):
    def __init__( self, myrepr = '', idx = None, meta_obj = None, meta_idx = None, annotations = None ):
        self.idx = idx
        self.meta_obj = meta_obj
        self.meta_idx = meta_idx
        self.myrepr = myrepr
        self.annotations = annotations

    def __repr__( self ):
        return '<indexed_object "{0}">'.format( self.myrepr.replace( "'", "\'" ))

    def index( self, idx = None):
        if idx:
            self.idx = idx
        return self.idx

    def split( self ):
        res = []
        if len( self.__repr__() ) > 0:
            for k in range( 0, len( self.__repr__() )):
                res.append( indexed_object( myrepr = self.__repr__()[ k ],
                                            idx = k,
                                            meta_obj = self,
                                            meta_idx = self.idx,
                                            annotations = self.annotations ))
            return res

# inverse function to method split() of class indexed_object
# this function is not used here (provided for the sake of symetry)
def contract( idx_obj_lst ):
    rep = ''
    prev_idx = -1
    metaobj = None
    for a in idx_obj_lst:
        assert( (prev_idx + 1) == a.idx )
        prev_idx = a.idx
        rep += a.__repr__()
        if metaobj is None:
            metaobj = a.meta_obj
        else:
            assert( metaobj is a.meta_obj )
    assert( metaobj.__repr__() == rep )  
    return metaobj

def text_to_indexed_obj_lst( txt, tokenizer = lambda s : s.strip().split( ' ' ) ):
    tokens = tokenizer( txt )
    res = []
    for k in range( 0, len( tokens )):
        res.append( indexed_object( tokens[ k ], k ))
    return res

# NOTE my idx symbols are in fact index python objects 

# first I must build the vocabulary of atoms of both streams to find a separator not already in that list
def find_unused_char( char_lst ):
    res = ord( '\u265E' ) # assert( res == "\N{BLACK CHESS KNIGHT} )"
    while chr( res ) in char_lst:
        res += 1
    return chr( res )

def char_vocab( a, b ):
    def filterString( c, lst ):
       if 'i' in lst:
          return False
       else:
          return True
    res = []
    all = a + b
    for o in all:
        res += filter( lambda  x  : filterString( x, res ), list( o.__repr__() ))
    return res

class aligned_indexed_object( object ):
    
    def __init__( self, obj = None, alignment = None, previous_obj = None, next_obj = None ):
        assert( type( obj ) is indexed_object )
        assert( type( algn_atom_lst ) is list )
        assert( type( alignment ) is dict )
        assert( type( dict[ 'aligned_atoms' ] is list ))
        for x in dict[ 'aligned_atoms' ]:
            assert( (type( x ) is tuple) and (len( x ) == 2) )
            assert( (type( x[ 0 ] is indexed_object)) and (type( x[ 1 ] is indexed_object)) )
        assert( type( dict[ 'aligned_atom_spans' ] is list ))
        for sp in dict[ 'aligned_atom_spans' ]:
            assert( type( sp ) is list )
            for x in sp:
                assert( type( x ) is indexed_object )
            
        for o in alignment:
            assert( type( o ) is indexed_object )
            
        self.alignment = alignment
        self.predecessor = previous_obj
        self.successor = next_obj

class alignment_info( object ):
    # the initial data structure is dissymetric since the reference stream (dim 0)
    # has been resegmented to have one reference symbol per token, and the
    # hypothesis stream (dim 1) has been updated accordingly to take into account
    # reference segmentation of the reference stream (dim 0).
    ref_dim = 0
    hyp_dim = 1
    
    def __init__( self, data = ([], []) ):
        self.data = data
        self.post_align()

    def size( self ):
        return len( self.data[ 0 ] )

    def ref_isymb( self, i ):
        return self.data[ alignment_info.ref_dim ][ i ][ 0 ] 

    def ref_matched_atom_idx_span_lst( self, i ):
        return self.data[ alignment_info.ref_dim ][ i ][ 1 ] 

    def hyp_matched_isymb_span_lst( self, i ):
        return self.data[ alignment_info.ref_dim ][ i ][ 2 ]

    def hyp_matched_atom_idx_span_lst( self, i ):
        return self.data[ alignment_info.ref_dim ][ i ][ 3 ]

    # ---- quality basic estimators

    def ref_size( self, i ):  # positive estimator
        return len( self.ref_isymb( i ).__repr__())

    def ref_realign_ratio( self, i ): # positive estimator
        mtchsz = 0
        for refspan in self.ref_matched_atom_idx_span_lst( i ):
            mtchsz += len( refspan )
        return  mtchsz / self.ref_size( i )

    def hyp_nbr_of_isymbs( self, i ): # a negative estimator
        hyp_isymb_cnt = 0
        prev_hyp_isymb = None
        for hyp_isymb_span in self.hyp_matched_isymb_span_lst(i):
            for hyp_isymb in hyp_isymb_span:
                if hyp_isymb != prev_hyp_isymb:
                    hyp_isymb_cnt += 1
        return hyp_isymb_cnt

    def median_of_unaligned_hyp_symbol_per_hyp_span( self, i ): # a negative estimator
        unaligned_hyp_isymb_atom_cnt_lst = []
        
        for k in range( 0, len( self.hyp_matched_isymb_span_lst( i ) )):
            total_matched_hyp_span_atom_cnt = sum( map( lambda s: len( s.__repr__()), self.hyp_matched_isymb_span_lst( i )[ k ] ))
            unaligned_hyp_atoms_cnt = total_matched_hyp_span_atom_cnt - sum( map( lambda l : len( l ), self.hyp_matched_atom_idx_span_lst( i )[ k ] ))
            unaligned_hyp_isymb_atom_cnt_lst.append( unaligned_hyp_atoms_cnt )

        if unaligned_hyp_isymb_atom_cnt_lst != []:
            res = median( unaligned_hyp_isymb_atom_cnt_lst )
            assert( res >= 0 )
        else:
            res = 0
        return res

    #----- end of quality estimators section 

    def print_ref( self ):
        # Note: completely unaligned hypothesis symbols are not printed here, because there would be no purpose
        # in doing so, since global symbol alignment is defined entirely by the levenshtein alignment algorithm
        # at the atomic level in the atom alignment matrix. Hypothesis symbols for which no atom have found a match
        # are of no interest, trying to undo (localy) this decision for some of them at this stage of processing
        # would be akin to redefining the global principle of alignment of the levenshtein algorithm.
        # But, trying to improve the reconstruction of the reference data in the hypothesis stream by force aligning
        # some symbols of the hypothesis which have found only a partial match at the atom level retains all its sense.
        
        for i in range( 0, self.size() ):
            msg = ''
            msg +=  'i= {0} ref="{1}" '.format( i, self.ref_isymb( i ) )
            msg += '\t ref atom matches per span: '
            mtchsz = 0
            for refspan in self.ref_matched_atom_idx_span_lst( i ):
                msg += '['
                for idx in refspan:
                    mtchsz += 1
                    msg += '{0}'.format( self.ref_isymb( i ).__repr__()[ idx ] )
                msg += '] '
            msg += 'i= {0} ref.% {1:1.2f} ={2:d}/{3:d} refsz={4:d}'.format( i,
                                                                            self.ref_realign_ratio( i ),
                                                                     mtchsz,
                                                                     self.ref_size(i),
                                                                     self.ref_size(i) )
            msg += '\n\thyp. isymb lists per span '
            for rng_k in range( 0, len( self.hyp_matched_isymb_span_lst( i ))):
                # k-th hyp match span
                msg += '[ ' 
                for hyp_isymb_z in range( 0, len( self.hyp_matched_atom_idx_span_lst( i )[ rng_k ])): 
                        # z-th hyp isymb in the k-th hyp match span
                        msg += '{0} '.format( self.hyp_matched_isymb_span_lst( i )[ rng_k ][ hyp_isymb_z ].__repr__())
                msg += ' ]'     
            msg += '\n\thyp. atom lists per span '
            for rng_k in range( 0, len( self.hyp_matched_isymb_span_lst( i ))):
                # k-th hyp match span
                msg += '[ ' 
                for hyp_isymb_z in range( 0, len( self.hyp_matched_atom_idx_span_lst( i )[ rng_k ])): 
                        # z-th hyp isymb in the k-th hyp match span
                        msg += '[ ' 
                        for hyp_x in self.hyp_matched_atom_idx_span_lst( i )[ rng_k ][ hyp_isymb_z ]:
                            msg += '{0}/{1:d} '.format( self.hyp_matched_isymb_span_lst( i )[ rng_k ][ hyp_isymb_z ].__repr__()[ hyp_x ],
                                                        hyp_x)                              
                        msg += ' ]' 
                msg += ' ]'                      
            msg += '\n\t# i= {0} matched hyp. symbols {1}, median unalign hyp atoms {2}'.format( i,
                                                                                                 self.hyp_nbr_of_isymbs( i ),
                                                                                                 self.median_of_unaligned_hyp_symbol_per_hyp_span( i ) )
            print( msg )
            mgs = ''
            
    def unalign( self, i = None ):
        assert( (i is not None) and (type( i ) is int) and (i >= 0) and (i < self.size()) )
        self.data[ alignment_info.ref_dim ][ i ][ 1 ].clear()
        self.data[ alignment_info.ref_dim ][ i ][ 2 ].clear()
        self.data[ alignment_info.ref_dim ][ i ][ 3 ].clear()
        
    def post_align( self ):
        # filter out unaligned portions to apply a final alignment refinement
        # using the quality estimators
        # values and tests have been empirically fixed for French/English on a small text sample.
        for i in range( 0, self.size() ):
            refsz = self.ref_size( i )
            if refsz < 3:
                if (self.ref_realign_ratio( i ) != 1.0) or (self.hyp_nbr_of_isymbs( i ) > 1):
                    self.unalign( i )
            else:
                if refsz < 7:
                    if (self.ref_realign_ratio( i ) < 0.66) or (self.hyp_nbr_of_isymbs( i ) > 3) or (self.median_of_unaligned_hyp_symbol_per_hyp_span( i ) > 3):
                        self.unalign( i )
                else:
                     if (self.ref_realign_ratio( i ) < 0.80) or (self.median_of_unaligned_hyp_symbol_per_hyp_span( i ) > 5):
                        self.unalign( i )
        
class dynprogalign:
        def __init__( self, a = None, b = None, comparator = None, noalign_tag = '*NOALGN*/None/[]/0.00' ):
            assert( (type( a ) is list) and (len( a ) > 0) )
            assert( False not in [ (type( x ) is indexed_object) for x in a ] )
            assert( (type( b ) is list) and (len( b ) > 0) )
            assert( False not in [ (type( x ) is indexed_object)  for x in b ] )
            self.a = a
            self.b = b
            self.sep = indexed_object( find_unused_char( char_vocab( a, b) ))
            if comparator is None:
                self.comparator = lambda x,y : x.__repr__() == y.__repr__()
            self.OK_COST = 0
            self.DELETION_COST = 1
            self.INSERTION_COST = 1
            self.SUBSTITUTION_COST = 2
            self.noalign_tag = noalign_tag
            self.atom_align_info = None
            self.isymb_align_info = None
            #self.debug = True
            self.debug = False

        def to_atoms( self, symlst ):
            atoms = [] ; max_s = 0
            for i in range( 0, len( symlst ) ):
                s = symlst[ i ] ; sln = len( s.__repr__() )
                if sln > max_s:
                   max_s = sln
                atoms += s.split()
                atoms.append( self.sep )
            atoms.pop()
            return ( atoms, max_s )

        def print_matrix( self, M ):
            for i in range( 0, len( M ) ):
                lntxt = ''
                for j in range( 0, len( M[ 0 ] )):
                    lntxt = lntxt + ' {0:3d}'.format(  + M[ i ][ j ] )
                print( lntxt )
            print( '=================' )
            print()

        def F( self, x, y ):
            return x + y

        def G( self, x, y, z ):
            return min( x, y, z )

        def build_matrix( self, x, y ):
            # this function builds the content of the Dynmic Programming Matrix
            # and calls the path extraction function.
            # x = ref symbol vector (one matrix line for each ref symbol)
            # y = hyp symbol vector (one matrix column for each hypothesis symbol)
            dummy = 0
            M = [ z[ : ] for z in [ [ dummy ] * (len( y ) + 1) ] * (len( x ) + 1)]
            
            if self.debug:
                self.print_matrix( M )

            assert( len( M ) == (len( x ) + 1) )
            assert( len( M[0] ) == (len( y ) + 1) )
            
            i = 0 ; j = 0;
            
            M[ 0 ][ 0 ] = 0;

            for i in range( 0, len( x )):
               M[ i + 1 ][ 0 ] = self.F( M[ i ][ 0 ], self.DELETION_COST ) #   c( r_i , epsilon )
      
            for j in range( 0, len( y )):
               M[ 0 ][ j + 1 ] = self.F( M[ 0 ][ j ], self.INSERTION_COST ) #   c( epsilon, h_i )
              
            for i in range( 0,  len( x )):
                 for j in range( 0, len( y )): 
                       match_mismatch_res = 0
                       if (len( x ) != 0) and  (len( y ) != 0) and (self.comparator( x[ i ], y[ j ]) is True):
                           match_mismatch_res = self.F( M[ i ][ j ], self.OK_COST )  # correct alignment case
                       else:
                           match_mismatch_res = self.F( M[ i ][ j ], self.SUBSTITUTION_COST ) # substitution case
                       M[ i+1 ][ j+1 ] = self.G( match_mismatch_res,
                                                 self.F( M[ i ][ j+1 ], self.DELETION_COST ),
                                                 self.F( M[ i+1 ][ j ], self.INSERTION_COST ) )
            # now M[i][j] holds the cost of aligning the sequence ref[i] with hyp[j]
            if self.debug:
                self.print_matrix( M )
            return M

        def diagonal_move( self, mv ):
           return mv == (1, 1)

        def horizontal_move( self, mv ):
           return mv == (0, 1)

        def vertical_move( self, mv ):
           return mv == (1, 0)

        def find_path( self, x, y, M ):
            pth = []
            pth_res = ''
            i = len( x ) ; j = len( y )
            mm = 0
            x_del = 0
            x_ins = 0
            x_sub_or_match = 0;
            pth.insert( 0, M[ len( x ) ][ len( y ) ] )

            # NOTE: going for the minimal cost right from the beginning
            # of the search will alig foo and fxo as
            # f/oo
            # fxo/

            # starting from the right down corner initial position there are
            # 3 possible moves back from the current position:
            #     x_ins (horizontal, x_del (vertical) and x_sub_or_match (diagonal)
            # Note: impossible move along the borders are encoded as maxint
            
            i = len( x ) ; j = len( y )
            while  (i > 0)  or (j > 0):
                mm = maxint
                
                if j > 0 :
                    x_del =  M[ i ][ j - 1 ]
                else:
                    x_del = maxint
                    
                if i > 0 :
                    x_ins = M[ i - 1 ][ j ]
                else:
                    x_ins = maxint
                    
                if (i > 0) and (j > 0):
                    x_sub_or_match =  M[ i-1 ][ j-1 ]
                else:
                    x_sub_or_match = maxint

                if self.debug:
                    print( 'cost_triplet (i, d, o/s) = {0}'.format( (x_ins, x_del, x_sub_or_match )))

                # find the total target minimal cost path
                mm = self.G( x_ins, x_del, x_sub_or_match )
                pth.insert( 0, mm )

                # choose a move with the target minimal cost path (there might be 2 possible moves
                # sometimes (choosing between insertion and deletion, but which ever I choose
                # it is indifferent because in the end all the alternative path in the matrix
                # resolve to the same global issue in term of alignment.

                if mm == x_sub_or_match:
                    if (j > 0) and (i > 0 ):
                        if (M[i][j] - M[i-1][j-1]) == self.OK_COST:
                            if self.debug:
                                print( 'found o' )
                            pth_res = 'o' + pth_res
                        else:
                            if self.debug:
                                print( 'found s' )
                            pth_res = 's' + pth_res
                        j -= 1
                        i -= 1
                        
                elif mm == x_ins:
                    # if we can make an insertion move we do it,
                    # although a delete move might be possible is if (x_del != maxint)
                    if( i > 0 ):
                        if self.debug:
                            print( 'found i' )
                        pth_res = 'i' + pth_res
                        i -= 1
                elif mm == x_del: # this is useful if a delete move is the only possible move with minimal cost
                    if( j > 0 ):
                         if self.debug:
                             print( 'found d' )
                         pth_res = 'd' + pth_res
                         j -= 1
                else:
                    break;

                if self.debug:
                    print()
                    print( 'pth= {0}'.format( pth ))
                    print( 'pth_res= {0}'.format( pth_res ))
                    answ = input( '======continue ? (other/n) ' )
                    if answ == 'n':
                        exit( 1 )
                    print()
                    
            res = 0;
            for k in range( 0, len( pth_res ) ):
              res += pth[ k ]
            return pth_res

        def levenshtein( self, a_toms, b_toms, max_w = 10 ):
            algn_atom_res = []   
            a_k = 0
            b_k = 0
            m = [] ;
            pth_res = '' ;
            first_run = True

            if self.debug:
                print( '================levenshtein' )
                print( '\ta_toms= {0}  \n\tb_toms= {1}  \n\tmax_w= {2} \n\tres= {3}'.format( a_toms, b_toms, max_w, algn_atom_res ))

            while first_run or (a_k < len( a_toms)) or (b_k < len( b_toms)):
                
                x = a_toms[ a_k : min( (a_k + max_w), len( a_toms )) ]
                y = b_toms[ b_k : min( (b_k + max_w), len( b_toms )) ]
                if self.debug:
                    print( '\n 0 §§§§§§§§§§§§§§§§§§§§§§§ a_k = {0}   b_k = {1}  \nx={2} \ny={3}'.format( a_k, b_k, x, y )) 
                m = self.build_matrix( x, y )
                
                pth_res = self.find_path( x, y, m )

                # build the word map from the back_path
                if self.debug:
                    print( '=======>> algn_path= {0}'.format( pth_res ))
                for mv in pth_res:
                    if mv == 'o':
                        algn_atom_res.append( (a_toms[ a_k ], b_toms[ b_k ] ) )
                        a_k += 1 ; b_k += 1
                    elif mv == 's':
                        algn_atom_res.append( (a_toms[ a_k ], None       ) )
                        algn_atom_res.append( (None       , b_toms[ b_k ]) )
                        a_k += 1 ; b_k += 1
                    elif mv == 'i':
                        algn_atom_res.append( (a_toms[ a_k ], None       ) )
                        a_k += 1
                    else:
                        assert( mv == 'd' )
                        algn_atom_res.append( (None       , b_toms[ b_k ]) )
                        b_k += 1

                if self.debug:
                    print( 'THE intermediary RES IS {0}'.format( algn_atom_res ))
                    print( '==================== end of matrix filling =======' )
                    print( '====> \ta_toms= {0}  \n\tb_toms= {1}  \n\tmax_wc= {2} \n\tres= {3}\n a_k= {4}\n b_k= {5}'.format( a_toms,
                                                                                                                          b_toms,
                                                                                                                          max_w,
                                                                                                                          algn_atom_res,
                                                                                                                          a_k,
                                                                                                                          b_k )) 
                first_run = False

                if a_k < len( a_toms ):
                    x = a_toms[ a_k :  min( (a_k + max_w), len( a_toms )) ]
                        
                if b_k < len( b_toms ):
                    y = b_toms[ b_k :  min( (b_k + max_w), len( b_toms )) ]

                if self.debug:
                    print( '\n 1 §§§§§§§§§§§§§§§§§§§§§§§ a_k = {0}   b_k = {1}  \nx={2} \ny={3}'.format( a_k, b_k, x, y )) 
            
                if (x != []) and (y == []):
                    # only a (reference) stream data remains, we add it to the result, so it is aligned with nothing
                    for t in a_toms[ a_k : ]:
                        algn_atom_res.append( ( t, None ) )
                            
                else:
                    if (x == []) and (y != []):
                        # align the remaining b stream objects with nothin since a stream reached its end
                        for t in b_toms[ b_k :  ]:
                            algn_atom_res.append( ( None, t ) )

                if self.debug:
                    print( '\n 2 §§§§§§§§§§§§§§§§§§§§§§§ a_k= {0} b_k= {1} \nx={2} \ny={3}'.format(a_k, b_k, x, y ))
                            
                # ------ end of:    while first_run or (a_leftover > 0) or (b_leftover > 0):
                
            if self.debug:
                print( '/////////////// END OF ALIGN')
                pprint( '\n THE RES IS {0}\n'.format( algn_atom_res ))

            return algn_atom_res

        def aligned_atom_span_to_text( self, algn_atom_lst ):
            res = '['
            for x in algn_atom_lst:
                if res != '[':
                    res += ', '
                res += '{0}_{1} '.format( x.__repr__(), x.idx )
            res = ']'
            return res

        def make_atom_span_correspondance( self, atom_map = None ):
            assert( type( atom_map ) is list )
            if self.debug:
                print( '\t==>>>  BEGINNING OF make_atom_span_correspondance()')
            for x in atom_map:
                assert( (type( x ) is tuple) and (len( x ) == 2) )
                if self.debug:
                    print( x )
                assert( (type( x[0]) is indexed_object) or (type( x[ 1 ] ) is indexed_object) )
            
            curr_atom_span = [] 
            atom_span_correspondance = []
            for x in atom_map:
               (a, b ) = x
               if (a is not None) or (b is not None ):
                   if (a == self.sep) or (b == self.sep):
                       atom_span_correspondance.append( curr_atom_span )
                       curr_atom_span = []
                   else:
                       curr_atom_span.append( (a, b) )
##            if self.debug:
##                print( 'A +++++>> atom_span_correspondance is' )
##                pprint( atom_span_correspondance )
##            if atom_span_correspondance == []:
##                atom_span_correspondance = [ atom_map ] # we have only a single symbol which constitutes the entirestream
            if self.debug:
                print( 'B +++++>> atom_span_correspondance is' )
                pprint( atom_span_correspondance )
            if curr_atom_span != []:
                atom_span_correspondance.append( curr_atom_span )
            if self.debug:
                print( 'RESULT +++++>> atom_span_correspondance is' )
                pprint( atom_span_correspondance )
            return atom_span_correspondance

        def atom_span_to_isymb( self, algn_atom_span, ref_dim = 0 ):
            if self.debug:
                print( 'atom_span_to_isym algn_atom_span = {0}'.format( algn_atom_span ) )
            # return the isymb/None associated to reference dimension of the the aligned atom span
            ref_isymb = None ; ref_atoms = [] # will hold only ref atoms whithout any None value
            ref_atom_match_idx_lst = []
            matching_hyp_isymb_lst = []
            matching_hyp_isymb_idx_lst2 = []
            for (a, b) in algn_atom_span:
                if ref_dim == 0:
                    ref = a ; hyp = b
                else:
                    assert( ref_dim == 1 )
                    ref = b ; hyp = a
                if ref is not None:
                    if hyp is not  None:
                        ref_atom_match_idx_lst.append( ref.idx )
                        assert( hyp.meta_obj  is not None )
                        if matching_hyp_isymb_lst == []:
                            matching_hyp_isymb_lst.append( hyp.meta_obj )
                            matching_hyp_isymb_idx_lst2.append( [ hyp.idx ] )
                        else:
                            if hyp.meta_obj !=  matching_hyp_isymb_lst[ -1 ]:
                                matching_hyp_isymb_lst.append( hyp.meta_obj )
                                matching_hyp_isymb_idx_lst2.append(  hyp.idx  )
                            else:
                                 matching_hyp_isymb_idx_lst2[ -1 ].append(  hyp.idx )
                    if ref_isymb is None:
                        assert( type( ref.meta_obj ) is indexed_object )
                        ref_isymb = ref.meta_obj
            if self.debug:
                print( 'XXXXXXXXX> atom_span_to_isymb( {0}, ref_dim= {1} )= ( {2} , {3}, {4}, {5} )'.format(  algn_atom_span,
                                                                                                              ref_dim,
                                                                                                              ref_isymb,
                                                                                                              ref_atom_match_idx_lst,
                                                                                                              matching_hyp_isymb_lst,
                                                                                                              matching_hyp_isymb_idx_lst2 ))
            return (ref_isymb, ref_atom_match_idx_lst, matching_hyp_isymb_lst, matching_hyp_isymb_idx_lst2 )

        def make_isymb_correspondance( self, atom_span_correspondance, ref_dim = 0 ): 
            assert( (ref_dim == 0) or (ref_dim == 1) )
            if self.debug:
                print( '===========================================DEBUG make_isymb_correspondance')
                pprint( atom_span_correspondance )
                print( '-----------' )
            # Note: the ref dimension (ref_dim) in the atom_span_correspondance is the one
            # which provides the symbol boundaries information for determining the symbols

            ref_isymb_correspondance = []
            prev_ref_isymb = None
            ( curr_ref_isymb , curr_ref_atom_match_idx_lst, curr_ref_matching_hyp_isymb_lst, curr_ref_matching_hyp_isymb_idx_lst2 ) = ( None , [], [], [] )
            
            for algn_atom_span in atom_span_correspondance:
                ( curr_ref_isymb, ref_atom_match_idx_lst, matching_hyp_isymb_lst, matching_hyp_isymb_idx_lst2 ) = self.atom_span_to_isymb( algn_atom_span, ref_dim )
                if self.debug:
                    print( '================ make_isym_correspondance for loop')
                    print( '\talgn_atom_span = {0}'.format( algn_atom_span ))
                    algn_atom_span_info  = ( curr_ref_isymb, ref_atom_match_idx_lst, matching_hyp_isymb_lst, matching_hyp_isymb_idx_lst2)
                    print( '\n\t( curr_ref_isymb, ref_atom_match_idx_lst, matching_hyp_isymb_lst, matching_hyp_isymb_idx_lst ) = {1}'.format( algn_atom_span_info ))
                    #==========================================================
                if curr_ref_isymb is not None:
                    if prev_ref_isymb is None:
                        prev_ref_isymb = curr_ref_isymb
                        ref_isymb_correspondance.append( (curr_ref_isymb, [], [], [] ) )
                        ref_isymb_correspondance[ -1 ][ 1 ].append( ref_atom_match_idx_lst )
                        ref_isymb_correspondance[ -1 ][ 2 ].append( matching_hyp_isymb_lst )
                        ref_isymb_correspondance[ -1 ][ 3 ].append( matching_hyp_isymb_idx_lst2 )
                        if self.debug:
                            print( 'appended first span in ref_isymb_correspondance = {0}'.format( ref_isymb_correspondance ))
                            pprint( ref_isymb_correspondance )
                            print( 'A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' )
                    else:
                        if curr_ref_isymb == prev_ref_isymb:
                            ref_isymb_correspondance[ -1 ][ 1 ].append( ref_atom_match_idx_lst )
                            ref_isymb_correspondance[ -1 ][ 2 ].append( matching_hyp_isymb_lst )
                            ref_isymb_correspondance[ -1 ][ 3 ].append( matching_hyp_isymb_idx_lst2 )
                            if self.debug:
                                print( 'B=========' )
                                pprint( ref_isymb_correspondance )
                                print( '==========' )
                        else:
                            prev_ref_isymb = curr_ref_isymb
                            ref_isymb_correspondance.append( (curr_ref_isymb, [], [], [] ) )
                            ref_isymb_correspondance[ -1 ][ 1 ].append( ref_atom_match_idx_lst )
                            ref_isymb_correspondance[ -1 ][ 2 ].append( matching_hyp_isymb_lst )
                            ref_isymb_correspondance[ -1 ][ 3 ].append( matching_hyp_isymb_idx_lst2 )
                            if self.debug:
                                print( 'C========' )
            if self.debug:
                print( '===========result ______________make_isymb_correspondance dim= {0}'.format( ref_dim ) )
                pprint( ref_isymb_correspondance )
            return ref_isymb_correspondance

        def merge_successor( isymb_correspondance, i ):
            assert( (type( i ) is int) and (type( isymb_correspondance ) is list) and ( i < len( isymb_correspondance ) - 1) )
            assert( isymb_correspondance[ i ][ 0 ] == isymb_correspondance[ i + 1 ][ 0 ] )
            isymb_correspondance[ i ][ 2 ] += isymb_correspondance[ i + 1 ][ 2 ]
            isymb_correspondance[ i ][ 3 ] += isymb_correspondance[ i + 1 ][ 3 ] 
            del isymb_correspondance[ i + 1 ]
        
        def retokenize( self, atom_map, force_realigner = 'keep', realignment_threshold = 1.0 ):
            # force_realigner can be either : 'previous', 'next', 'discard', 'keep'.
            if self.debug:
                print( 'retokenize atom map is {0}'.format( atom_map ))
           
            atom_span_correspondance = self.make_atom_span_correspondance( atom_map = atom_map )
            a_isymb_correspondance = self.make_isymb_correspondance( atom_span_correspondance, ref_dim = 0 )
            if self.debug:
                print( 'retokenize >>> a_isymb_correspondance=' )
                pprint( a_isymb_correspondance )
            
            b_isymb_correspondance = self.make_isymb_correspondance( atom_span_correspondance, ref_dim = 1 )
            if self.debug:
                print( 'retokenize>>> b_isymb_correspondance= ' )
                pprint( b_isymb_correspondance )

            i = 0
            align_quality = []
            while i < (len( a_isymb_correspondance ) - 1):
              curr_ref_isymb = a_isymb_correspondance[ i ][ 0 ]
              if curr_ref_isymb == a_isymb_correspondance[ i+1 ][ 0 ]:
                  self.merge_successor( a_isymb_correspondance, i )
                  self.merge_successor( b_isym_correspondance, i )
              else:
                  i += 1
            return (a_isymb_correspondance, b_isymb_correspondance)

        def print_align_info( self ):
            if self.debug:
                print( '_____ indexed symbol correspondance _____' )
                print( 'a stream symbol\ta stream atom alignment\tb stream atom alignment' )
            (a_symb_algn, b_symb_algn) = self.isymb_align_info
            i = 0 ;  j = 0
            msg = ''
            while (i < len( a_symb_algn )) or (j < len( b_symb_algn )):
                if i < len( a_symb_algn):
                    ( a_isymb, matching_a_idx_lst, matching_b_isymb_lst, matching_b_isymb_idx_lst2 ) = a_symb_algn[ i ]
                    msg += '{0}\t{1}\t{2}\t{3}\n'.format( a_isymb, matching_a_idx_lst, matching_b_isymb_lst, matching_b_isymb_idx_lst2 )
                    print( msg ) ; msg = ''
                    i += 1
                # for debug only
                else:
                    break

        def levenshtein_map( self ):
           if len( self.a ) == 0:
               dummy = 0
           elif len( self.b ) == 0:
               dummy = 0
           else:
              (a_toms, max_aln) = self.to_atoms( self.a )
              (b_toms, max_bln)  = self.to_atoms( self.b )
              self.atom_align_info = self.levenshtein( a_toms, b_toms, int( max( len( self.a), len( self.b ) * max( max_aln, max_bln ) * 1.20 )) )
              if self.debug:
                  print( 'atom map is {0}'.format( self.atom_align_info ))
              self.isymb_align_info = self.retokenize( self.atom_align_info )
              if self.debug:
                  print( '_______levenshtein_map()________RESULT symb_align info' )
                  self.print_align_info()
              return self.isymb_align_info
            
#----- end of class dynprogalign
            
def demo( ):
    ##answ = input( 'proceed with second test ? (y/o) ' )
    ##if answ != 'y':
    ##     exit( 1 )
    ##print( '_______________test 0 several symbols with all possible cases' )
    #------------------------------------------------
    ##aligner = dynprogalign( [ indexed_object( 'alpha', 0 ), indexed_object( 'beta', 1 ), indexed_object( 'gamma', 2 ) ],
    ##                        [ indexed_object( 'al', 0), indexed_object( 'pha', 1 ), indexed_object( 'be', 2 ), indexed_object( 'gamma', 3 ) ] )
    ##print( '_______________test 0 equality (on symbol only in ref and hyp' )
    ##aligner = dynprogalign( [ indexed_object( 'alpha', 0 ) ], [ indexed_object( 'alpha', 0 )] )
    ##result = aligner.levenshtein_map( )
    ##pprint( result )
    ##answ = input( 'proceed with second test ? (y/o) ' )
    ##if answ != 'y':
    ##     exit( 1 )
    ##print( '_______________test 1 no match (ref symbol longer than hyp symbol)' )
    ##aligner = dynprogalign( [ indexed_object( 'alpha', 0 ) ], [ indexed_object( 'xi', 0 )] )
    ##result = aligner.levenshtein_map( )
    ##pprint( result )
    ##answ = input( 'proceed with third test ? (y/o) ' )
    ##if answ != 'y':
    ##     exit( 1 )
    ##print( '_______________test 2 no match (ref symbol shorter than hyp symbol)' )
    ##aligner = dynprogalign( [ indexed_object( 'mi', 0 ) ], [ indexed_object( 'beta', 0 )] )
    ##result = aligner.levenshtein_map( )
    ##pprint( result )
    ##answ = input( 'proceed with fourth test ? (y/o) ' )
    ##if answ != 'y':
    ##     exit( 1 )
    ##print( '_______________test 3 several symbols ' )
    ###aligner = dynprogalign( text_to_indexed_obj_lst( 'il arrive en retard.' ), text_to_indexed_obj_lst( 'il part en retard'))
    ##aligner = dynprogalign( text_to_indexed_obj_lst( 'il arrive' ), text_to_indexed_obj_lst( 'hier elle arrivait'))
    ##result = aligner.levenshtein_map( )
    ##pprint( result )
            
    ##answ = input( 'proceed with fifth test ? (y/o) ' )
    ##if answ != 'y':
    ##        exit( 1 )
    ##print( '_______________test 4 several symbols with all possible cases' )
    ##text1 = 'il arrive en retard avec dans sa poche un petit discours.'
    ##text2 = 'hier, elle arrivait sans aucun mouchoir dans sa poche, ni discours non plus.'
    ##print( 'aligning:\ntext1:\t{0}\nwith\n text2:\t{1}\n'.format( text1, text2 ))
    ##    aligner = dynprogalign( text_to_indexed_obj_lst( text1 ),
    ##                            text_to_indexed_obj_lst( text2 ))
    ##    result = aligner.levenshtein_map()
    ##    pprint( result )

    ##answ = input( 'proceed with sixth test ? (y/o) ' )
    ##if answ != 'y':
    ##     exit( 1 )
    ##print( '_______________test 5 several symbols with all possible cases' )
    ##aligner = dynprogalign( text_to_indexed_obj_lst( 'arrive' ),
    ##                        text_to_indexed_obj_lst( 'arrivait'))
    ##result = aligner.levenshtein_map()
    ##pprint( result )

    answ = input( 'proceed with seventh test ? (y/o) ' )
    if answ != 'y':
         exit( 1 )
    print( '_______________test 6 several symbols with all possible cases' )
    aligner = dynprogalign( text_to_indexed_obj_lst( 'Alpha beta gamma' ),
                            text_to_indexed_obj_lst( 'alpha'))
    result = aligner.levenshtein_map()
    if aligner.debug:
        pprint( result )
        print( 'done' )
        #-----------------------------------
    r = alignment_info( data = result )
    print( '\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n' )
    r.print_ref()


#demo()


