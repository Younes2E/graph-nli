## MIT License
##
## Copyright (c) 2019 aakorolyova
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## opies or substantial portions of the Software.
##
## HE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
## Note: A large part of this file comes initially from (2019):
## https://github.com/aakorolyova/DeSpin/blob/master/text_span.py
## with additions in 2020 by Maha BOUZAIENE <bouzaiene.maha@gmail.com>
## and Patrick PAROUBEK <pap@limsi.fr>
## Updated for dependency based core noun group extraction by Patrick Paroubek on April 05 2022

import sys
import glob
import codecs
import os
import collections
import io
from copy import deepcopy
import re

from functools import partial
from collections import defaultdict

from pprint import pprint

import csv

#import pdb; pdb.set_trace()

from itertools import permutations, count
from functools import reduce

#DEBUG = True
DEBUG = False

DEBUG_SHORT_REP_ANNOTS = True
#DEBUG_SHORT_REP_ANNOTS = False

ALLINSTANCES = {}

# utility function for forward compatibility, if using unification library
def name( x ):
    return x.name

def varify( obj ):
    nm = name( obj )
    if nm in ALLINSTANCES:
        if ALLINSTANCES[ nm ] == obj:
            pass
        else:
            sys.stderr.write( 'ERROR name instance clash in varify() name is: "{0}", for object: {1}, of type: {2}\n'.format( nm, obj, type( obj ) ))
            raise ValueError
    else:
       ALLINSTANCES[ nm ] = obj

def getobj( nm ):
    if nm not in ALLINSTANCES.keys():
        return None
    else:
        return ALLINSTANCES[ nm ]

#----------- 
    
class mwu( object ):
    #--- class variables 
    default_mwu_name_prefix = 'MWU'
    default_mwu_cnt = count( start = 0, step = 1 ) 
    # txsps is a list of pairs of character offsets e.g.  mwu( nm='MWU_4', txtsps=[ (13300 , 13320)])
    # example of mwu structure:
    # m1=mwu( nm='MWU_4', txtsps=[ (13300 , 13320)], annotations={'comments':'ceci est un test', 'POS_GRACE':'Ncms'})
    
    def __init__( self, nm = None, txtsps = None, typ = None, annotations = None, idx = 0 ):
        #--- name.
        if( nm is None ):
            self.inst_cnt = next( mwu.default_mwu_cnt )
            self.name = mwu.default_mwu_name_prefix + '_' + '{0}'.format( self.inst_cnt );
        else:
            self.name = nm
        if( txtsps is None ):
             self.txtspans = []
        else:
            self.txtspans = txtsps
        #--- annotations global to the mwu (attribute value associations in a dictionary).
        if( annotations is None ):
            self.annotations = {}
        else:
            self.annotations = annotations
        if( typ is None ):
             self.typ = ''
        else:
            self.typ = typ
        varify( self )

    def text_n( self, doc, n ):
        tsp = self.txtspans[ n ]
        assert( type( tsp ) is tuple )
        return( doc.span( tsp[0], tsp[1] ))
                
    def text( self , doc, sep = None ):
        res = ''
        for i in range( 0, len( self.txtspans ) ):
                if( sep is None ):
                    res += self.text_n( doc, i )
                else:
                    if( i > 0 ):
                        res += sep
                    res += self.text_n( doc, i )  
        return( res )

    def span_n( self, n ):
        if( n < len( self.txtspans) ):
            return( self.txtspans[ n ] )
        else:
            return( None )

    def __repr__( self ):
        res = 'mwu( nm=' + name( self ).__repr__() +  ', txtsps=['
        n = 0
        for sp in self.txtspans:
            res += sp.__repr__()
            ##            res += 'wnts( \'SPAN\', ' +  str( sp[0] ) + ',' + str( sp[1] ) + ')'
            n += 1
            if( n < len( self.txtspans )):
                res += ', '
        res +=  ']'
        res += ', typ= {0}'.format( self.typ.__repr__() )
        if self.annotations:
            res += ', annotations={0}'.format( self.annotations.__repr__() )
        if DEBUG_SHORT_REP_ANNOTS:
            res = res[ 0: min(200, len( res)) ] + ' ETC...'
        res += ')'
        return( res ) 

    def __str__( self ):
        res = '---mwu:\n\tname= ' + str( name( self ) )
        res += '\n\ttextspans is: ' + str( self.txtspans )
        res += '\n\ttyp is: ' + str( self.typ )
        if DEBUG:
            res += '\n\tannotations is: ' + str( self.annotations )[0:min(200, len( str( self.annotations )))] + ' ETC...'
        else:
            res += '\n\tannotations is: ' + str( self.annotations )
        res += '\n'
        return( res )

    def size( self ):
        return( len( self.txtspans ) )
    
         #---- annotations of the whole mwu)
    def set_annot( self, aspect, value ):
        self.annotations[ aspect ] = value

    def get_annot( self, aspect ):
        return( self.annotations[ aspect ] )

    def append( self, m ):
        # append all the text spans of m to the list of self.txtspans
        # and incorporate annotations of m onto self.annotations
        if len( m.txtspans ) > 0:
            for s in m.txtspans:
                self.txtspans.append( deepcopy( s ))
            self.annotations.update( m.annotations )
        return self

    def overlap( self, mwu2 ):
        if self.txtspans[0][0] < mwu2.txtspans[0][0]:
            first = self
        else:
            first = mwu2
        if self.txtspans[-1][-1] > mwu2.txtspans[-1][-1]:
            last = self
        else:
            last = mwu2
        return (first == last) or (first.txtspans[-1][-1] > last.txtspans[0][0])
            
        
#--- end of mwu class

class rel( object ):
    #--- class variables 
    default_rel_name_prefix = 'REL'
    default_rel_cnt = count( start = 0, step = 1 ) 
    def __init__( self, nm = None, src = None, trg = None, typ = None, idx = 0 ):
        assert( (type( nm ) is str) or (nm  is None))
        assert( type( src is mwu) or (type( src ) is relation))
        assert( type( trg is mwu) or (type( trg ) is relation))
        assert( type( typ is str) )
        self.inst_cnt = next( rel.default_rel_cnt )
        if( nm is None ):
            self.name = rel.default_rel_name_prefix + '_' + '{0}'.format( self.inst_cnt );
        else:
            self.name = nm
        self.src = src 
        self.trgt = trg 
        self.typ = typ

    def __repr__( self ):
        res = 'rel: nnm= ' + str(name( self )) + '\n\ttyp= ' +  str(self.typ) + '\n\tsrc= ' +  str(self.src)  + '\n\ttrg= ' + str(self.trgt) + '\n'
        return( res )

    def __str__( self ):
        sout = 'rel: nnm= ' + str(name( self )) + '\n\ttyp= ' +  str(self.typ) + '\n\tsrc= ' +  str(self.src)  + '\n\ttrg= ' + str(self.trgt) + '\n'
        return( sout )

    
class relation( rel ):
    def __init__( self, nm=None, src=None, trg=None, annotations = None, typ = None ):
         assert( (type( nm) is str) or (nm is None))
         assert( (type( src ) is mwu) or (type( src ) is relation) )
         assert( (type( trg ) is mwu) or (type( trg ) is relation) )
         assert( (type( typ ) is str) )
         if( annotations is None ):
             self.annotations = {}
         else:
             self.annotations = annotations
                 
         rel.__init__( self, nm,  src, trg, typ )
         varify( self )

     #---- annotations of the whole mwu)
    def set_annot( self, aspect, value ):
        self.annotations[ aspect ] = value

    def get_annot( self, aspect ):
        return( self.annotations[ aspect ] )

    def __repr__( self ):
        res = 'rel: nnm= ' + str(name( self )) + '\n\ttyp= ' +  str(self.typ) + '\n\tsrc= ' +  str(self.src)  + '\n\ttrg= ' + str(self.trgt) + '\n'
        if self.annotations:
            res += 'annotations=' + self.annotations.__repr__()
        res += ')'
        return( res )

    def __str__( self ):
        sout = 'rel: nnm= ' + str(name( self )) + '\n\ttyp= ' +  str(self.typ) + '\n\tsrc= ' +  str(self.src)  + '\n\ttrg= ' + str(self.trgt) + '\n'
        return( sout )
    

#--- end of class relation    

class construction( object ):
    # a construction is a (possibly empty) set of mwus and a (possibly empty) set of relations
    # with the condition that one of these two sets is not empty.
    # In addition there are two sets of optional mwus and optional relations that may be empty,
    # note that both of them can be empty.
    #--- class variables 
    default_construction_name_prefix = 'KSTR'
    default_construction_cnt = count( start = 0, step = 1 )
    
    def __init__( self, nm = '', typ = '',  mwus = {}, rels = {}, opt_mwus = {}, opt_rels = {}):
        assert(  (type( mwus ) is list) or (type( mwus) is dict) )

        self.inst_cnt = next( construction.default_construction_cnt )
        if nm == '':
            self.name = self.default_construction_name_prefix + '_' + '{0}'.format( self.inst_cnt )
        else:    
            self.name =  nm
        self.typ      = typ
        self.mwus  = mwus
        self.rels  = rels
        self.opt_mwus = opt_mwus
        self.opt_rels = opt_rels

        if type( mwus ) is list:
            assert( False not in [((type( m ) is mwu)) for m in mwus ])
            for e in mwus:
                self.mwus[ name( e ) ] = e
        else:
            assert( type( mwus ) is dict)
            assert( False not in [ (type( m ) is mwu) for m in mwus.values() ])
            self.mwus = mwus
                
        if type( rels ) is list:
            assert( False not in [ (type( m ) is rel) for m in rels ])
            for e in rels:
                self.rels[ name( e ) ] = e
        else:
            assert( type( rels ) is dict)
            assert( False not in [ (type( m ) is relation) for m in rels.values() ])
            self.rels = rels
        # warning: do not call varify() from here  since construction is not a terminal class in the inheritance graph.

    def __repr__( self ):
        sout = 'construction( nm= ' +  name( self ).__repr__() + ', rels=['
        n = 0
        for r in self.rels:
            sout += r.__repr__()
            n += 1
            if( n < len( self.rels ) ):
                sout += ', '
        sout += ']'
        return( sout )  

    def __str__( self ):
        sout = 'construction: name= ' + str( name( self )) + ' rels=['
        for r in self.rels:
            sout += r.__str__()
        sout += ']'
        sout += '\n'
        return( sout )
    
    #--- adding elements

    def add_elt_to_store( self, e, store = None ):
        assert( store in [ 'mwus', 'rels' ] )
        assert( (type( e ) is mwu) or (type( e ) is relation) )
        if type( e ) is mwu:
            self.mwus[ e.name ] = e
        elif type( e ) is relation:
            self.rels[ e.name ] = e
        else:
            sys.stderr.write( 'ERROR element {0} has a wrong type {1} for being added to construction {2}'.format( e, type( e ), self.name ))
            assert( 0 )

    def add_mwu( self, m ):
        self.add_elt_to_store( m, 'mwus' )

    def add_rel( self, r ):
        self.add_elt_to_store( r, 'rels' )

    #--- getting elements

    def get_mwu( self, k = None ):
        assert( k is not None)
        if k in self.mwus.keys():
            return self.mwus[ k ]
        else:
            return None

    def find_mwu_containing_span( self, s ):
        assert( (type( s ) is tuple) and (len(s) == 2) and (type(s[0]) is int) and (type(s[1]) is int) and (s[0] < s[1]) )
        for (nm, mw) in self.mwus.items():
                if( mw.contains_span_p( s ) ):
                    return( mw )
        return( None )

    def find_mwus_with_type( self, typ ):
        res = []
        for m in self.mwus:
            if m.typ == typ:
                res.append( m )
        return res

    #--- removing elements

    def remove_element( self, e, store ):
        assert( (type( store ) is dict ) and ((type( e ) is mwu) or (type( e ) is list) or (type( e ) is dict) ) )
        if type( e ) is str:
            if e in self.store.keys():
                del self.store[ e ]
            else:
                pass
        else:
            if e.name in store:
                del self.store[ e.name ]
            else:
                pass
            
    def remove_mwu( self, m ):
        assert( (type( m ) is str ) or (type( m ) is mwu) )
        self.remove_element( m, self.mwus )

    def remove_rel( self, r ):
        assert( (type( r ) is str ) or (type( r ) is rel) )
        self.remove_element( r, self.relations )

    #------ optional part -----------

    def add_mwu_opt( self, m ):
        assert( (type( m ) is mwu) or (type(m) is list) or (type(m) is dict) )
        if( type( m ) is list ):
            assert( False not in [ (type( x ) is mwu) for x in m ])
            for e in m:
                self.mwus_opt[ e.name ] = e
        elif type( m ) is dict:
            assert( False not in [ (type( x ) is mwu) for x in m.values() ])
            self.mwus_opt.update( m )
        else:
            assert( type( m ) is mwu )
            self.mwus_opt[ m.name ] = m
        
    def add_rel_opt( self, r ):
        assert( (type( r ) is relation) or (type(r) is list) or (type( r) is dict ) )
        if( type( r ) is list ):
            for e in r:
                self.rels_opt[ e.name ] = e
        elif type( r ) is dict:
            self.rels_opt.update( r )
        else:
            assert( type( r ) is relation )
            self.rels_opt[ r.name ] = r
            
#----- end of class construction

def mwu_is_src_of_rel_typs( m, reltyps, rels ):
    for r in rels:
        if (r.typ in reltyps) and (r.src == m):
            return True
    return False
   
def pattern_filter_noun_group( doc, exclude_determiner_p = True ):
    STANZA_GN_SEED_RELS_EXCLUSION_LST     = [ 'compound', 'compound:prt', 'flat', 'goeswith', 'name' ]
    STANZA_GN_SUBTREE_RELS_EXCLUSION_LST  = [ 'cc', 'cc:preconj', 'conj', 'sconj', 'punct', 'mark', 'appos', 'nmod', 'acl:relcl', 'obl', 'aux',
                                              'punct', 'nsubj', 'csubj', 'nsubjpass',
                                              'csubjpass', 'obj', 'ccomp', 'xcomp', 'iobj', 'nmod:tmod', 'foreign', 'acl', 'acl:recl', 'appos',
                                              'neg', 'case', 'list', 'dislocated', 'parataxis', 'orphan', 'reparandum', 'dep', 'root',
                                              'vocative', 'discourse', 'expl', 'aux', 'aux:pass', 'cop' ]
    if exclude_determiner_p:
        STANZA_GN_SUBTREE_RELS_EXCLUSION_LST += [ 'det', 'det:predet' ]
    # we want to consider in the nominal subtree the following dependencies:
    # ---------relations where the target is either the core noun or an outer border word
    # compound, compound:prt, flat, goeswith
    # amod, advmod (when modifying an adjective),
    # conj (when the source and the head are adjectives or adverbs),
    #-------------------------
    # NOTE: This cases below are not handled for now
    # 'nmod:poss' only when the source is a possessive determiner
    # --------- relations where the source is either the core noun or an outter border word
    ## 'nmod:npmod' only when the source is a noun (exclude pronouns)
    ## NOTE: stanza annotates nmod:npmod as obl:npmod
    # 'nummod' 
    ## 'name' relation seems to exist in udped, is it produced by stanza ?
    #-----------

    # works with Stanza parser
    if DEBUG:
        print( 'processing doc== {0}'.format( doc.name ))
    # ---- step0 finding all noun words
    # select all words with upos NOUN
    noun_mwu_lst = []
    for (mwid, mw) in doc.mwus.items():
        #print( 'mw.annotations.keys() == {0}'.format( list(mw.annotations.keys()) ))
        msg = 'mw.annotations.values() == {0}'.format( list(mw.annotations.values()))
        # print( msg )
        if DEBUG:
            print( msg[0:min(200, len( msg ))] )
            print( 'DEBUG++++++++++++++++ {0}'.format( mw.annotations ))
            print( '\t\tmwu.typ== {0} mwu.nm== {1}'.format( mw.typ, mw.name ))
            if mw.typ == 'Token':
                if  mw.annotations[ 'w_upos' ] in [ ['NOUN'], ['PROPN'] ]:
                    noun_mwu_lst.append( mw )
    noun_mwu_lst_nm = [ x.name for x in noun_mwu_lst ]

    # remove all mwus which are source from a compound dependency
    core_noun_mwu_lst = []
    for (rid, r ) in doc.rels.items():
        if (r.src.name in noun_mwu_lst_nm) and (r.typ in [ 'compound' ]):
            pass
        else:
            if r.trgt.name in noun_mwu_lst_nm:
                core_mwu = doc.mwus[ r.trgt.name ]
                if core_mwu not in core_noun_mwu_lst:
                    core_noun_mwu_lst.append( core_mwu )
    
    if DEBUG:
        print( 'BEFORE expanding SUBTREES  noun word list ==  {0}'.format( [ x.text( doc ) for x in noun_mwu_lst ] ))

    # extract the dependency subtree associated to each core noun mwu and 
    # create one construction for each noun group seed word
    noun_group_res_lst = []  # one pattern is created associated to each noun token found previously
    
    for m in core_noun_mwu_lst:
        depth = 0
        if not mwu_is_src_of_rel_typs( m, STANZA_GN_SEED_RELS_EXCLUSION_LST, doc.rels.values() ):
            # NOTE: better to always initialize explicitely default parameter values in the call to avoid sometimes strange side effects.
            m_kstruct = construction( nm = 'noun_subtree_' + name( m ),
                                      typ = 'coreNG',
                                      mwus = {},
                                      rels = {},
                                      opt_mwus = {},
                                      opt_rels = {} )
            m_kstruct.add_mwu( m )
            if DEBUG:
                print( '>>>ADDING CORE MWU: {0} with construction {1}'.format( m, name( m_kstruct ) ) )
                print( 'm_kstruct.keys()=={0}'.format( m_kstruct.rels ) )
            outer_border = [] ; new_outer_border = []
            for r in doc.rels.values():
                if (r.typ not in STANZA_GN_SUBTREE_RELS_EXCLUSION_LST ) and (r.trgt == m):
                    # 'nmod:poss' only when the source is a possessive pronoun
                    # 'nmod:npmod' only when the source is a noun (exclude pronouns)
                    if DEBUG:
                        print( '>>>>>>>>ADDING OUTER_BODER relation {0}'.format( r ))
                    outer_border.append( r )
                    m_kstruct.add_mwu( r.src )
                    m_kstruct.add_rel( r )
            new_outer_border = []
            while outer_border:
                if DEBUG:
                    print( 'depth=={0}'.format( depth ))
                    print( 'outer_border== {0}'.format( outer_border ))
                    print( 'new_outer_border== {0}'.format( new_outer_border ))
                    print( '===========')
                for r_out in outer_border:
                    for r in doc.rels.values():
                        if (r.typ not in STANZA_GN_SUBTREE_RELS_EXCLUSION_LST ) and (r.trgt == r_out.src):
                            # 'nmod:poss' only when the source is a possessive pronoun
                            # 'nmod:npmod' only when the source is a noun (exclude pronouns)
                            if DEBUG:
                                print( '>>>>>>>>ADDING NEW OUTER_BODER relation {0}'.format( r ))
                            new_outer_border.append( r )
                            m_kstruct.add_mwu( r.src )
                            m_kstruct.add_rel( r )
                outer_border = new_outer_border  # pop the outer_border old level and append the new level
                new_outer_border = []
                depth += 1
                
            #------

            if DEBUG:
                print( 'A_____________________\n')
                print( '\n SUBTREE for mwu== {0} with text=={1} is:\n'.format( name( m ), m.text( doc ) ))
                print_rels( m_kstruct.rels )
                print( 'B_____________________\n')

            noun_group_res_lst.append( m_kstruct )
    if DEBUG:
        print( 'noun_group_res_lst is=={0}'.format( noun_group_res_lst ))
    return noun_group_res_lst


def print_rels( rels ):
    print( '========================= rels =============' )
    for (rnm, r) in rels.items():
        print( 'relation name={0} type={1} src={2}  {3} ==> trgt={4}  {5}'.format( r.name,
                                                                                   r.typ,
                                                                                   r.src.name,
                                                                                   r.src.annotations ['text'],
                                                                                   r.trgt.name,
                                                                                   r.trgt.annotations ['text'] ))
    print( '===================================================' )
    
def print_mwus( mwus, doc ):
    print( '========================= mwus =============' )
    for (mnm, m) in mwus.items():
        #print( 'mwu text== {0} name={1} type={2} annotations= {3}'.format( m.text( doc ), m.name, m.typ, m.annotations))
        if 'pos' in m.annotations.keys():
            print( 'mwu text== {0} pos== {1}'.format( m.text( doc ), m.annotations[ 'pos' ] ))
        else:
            print( 'no pos in annotations of mwu.name={0} mwu.text()=={1}'.format( m.name, m.text( doc ) ))
    print( '===================================================' )


class document( construction ):
    default_doc_name_prefix = 'D'
    default_doc_cnt = count( start = 0, step = 1 )
    
    def __init__(self, nm='', content='', metadata='', multi_word_units={}, relations={}, constructions={} ):
        self.inst_cnt = next( document.default_doc_cnt )
        
        if nm == '':
            nm  = document.default_doc_name_prefix + '_' + str( self.inst_cnt )
        else:
            pass
        self.name =  nm

        if( type(content) is not str ):
            raise ValueError
        else:
            self.ctnt = content
            
        if( type(metadata) is not str ):
            raise ValueError
        else:
            self.metadata = metadata

        self.mwus = {}
        self.rels = {}
        
        self.constructions = constructions
        # potentialy a document has multi-word units (mwus) annotations, relations (between two mwus, a source and a target)
        # and constructions (sets of relations)
        super(document, self).__init__( nm = nm, typ = 'document', mwus = multi_word_units, rels = relations )

    def len( self ):
        return( len( self.ctnt ) )

    def byte_len( self ):
        return( len( self.ctnt.encode('utf8') ))

    def content( self ):
        return( self.ctnt )

    def span( self, first, last ):
        return( self.ctnt[ first:last ] )

    def __repr__( self ):
        res = 'document( ident=\'' + self.name + '\''
        res += '\t\n, content=' + self.ctnt.__repr__()
        res += '\t\n, metadata=' + self.metadata.__repr__()
        res += '\t\n, multi_word_units = ['
        i=0
        for m_nm in self.mwus.keys():
            if( i > 0 ):
                res += ', '
            i += 1
            res += self.mwus[ m_nm ].__repr__()
        res += ']\n'
        res += '\t, relations= ['
        i = 0
        for r_nm in self.rels.keys():
            if( i > 0 ):
                res += ', '
            i += 1
            res += self.rels[ r_nm ].__repr__()
        res += ']\n'
        res += '\t, constructions= ['
        i = 0
        for k_nm in self.constructions:
            if( i > 0 ):
                res += ', '
            i += 1
            a_kstruct = self.constructions[ k_nm ]
            if DEBUG:
                print( '\t\ttype( a_kstruct )== {0} a_kstruct.name == {1}'.format( type( a_kstruct), a_kstruct.name ))
                print( '\t\ta_kstruct.__repr__()== {0}'.format( a_kstruct.__repr__() ))
            res = res + a_kstruct.__repr__()
        res += '] )\n'
        return( res )

    def __str__( self ):
        res = 'document is:' + '\tid= ' + str( self.name )
        res += '\tmetadata= ' + str( self.metadata )
        res += '\tlen(self.ctnt)= ' + str( len( self.ctnt ))
        res += '\tlen(mwus)=' + str( len(self.mwus) )
        res += '\tlen(rels)=' + str( len(self.rels) )
        return( res )

    #------- file io

    def read_text( self, path ):
        assert( (type( path ) is str) and (path != '') )
        buffer_sz = 100
        in_strm = codecs.open( path, mode='r', encoding='utf-8', errors='strict', buffering = -1)
        assert( in_strm )
        in_data = u''
        for l in in_strm.readlines( buffer_sz ):
            in_data += l
        self.ctnt = self.ctnt + in_data
        in_strm.close()

    def write_text( self, path ):
        assert( (type( path ) is str) and (path != '') )
        out_strm = codecs.open( path, mode='w', encoding='utf-8', errors='strict', buffering = -1)
        assert( out_strm )
        out_strm.write( self.ctnt )
        out_strm.close()


##    # pap 20210813
    def export_mwus_bert_to_csv( self, out_strm, headerp = False ):
        ## export mwus, their Stanza POS information and distilled BERT embeddings
        if DEBUG:
            print( '[[[[[[[[[[[preparing to export {0} mwus bert to csv'.format( len( self.mwus.keys() ) ))
        if headerp:
            header = '\t'.join( [ '#docid', 'mwu_name', 'mwu_type', 'mwu_text_spans', 'mwu_forms', 'w_lemma' , 'w_upos', 'w_xpos', 'w_feats'] )
            for i in range( 0, 128 ):
                header += '\tBERT{0}'.format( i )
            out_strm.write( header )
            out_strm.write( '\n' )
        for m_nm in self.mwus.keys():
            m = self.mwus[ m_nm ]
            if m.typ == 'Token':
                out_strm.write( self.name )
                out_strm.write( '\t' )
                out_strm.write( self.mwu_to_csv_str( m ) )
                if 'BERT_Representation' in m.annotations.keys():
                    #out_strm.write( 'BERT_Representation={0}\t'.format( m.annotations[ 'BERT_Representation' ] ))
                    for x in m.annotations[ 'BERT_Representation' ]:
                        out_strm.write( '\t{0}'.format( x ) )
                out_strm.write( '\n')
        out_strm.flush()
        if DEBUG:
            print( ']]]]]]]]]]]] done export {0} mwu bert to csv'.format( len( self.mwus.keys() ) ))

            
#--- end of class document


          


