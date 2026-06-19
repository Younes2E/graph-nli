import sys

from pprint import pprint

import spacy

from spacy import displacy     

from dynprogalign import indexed_object, dynprogalign, alignment_info

from text_span_v2 import mwu, relation, document, construction, mwu_is_src_of_rel_typs, name, print_mwus, print_rels, varify, getobj, name

from ntree import ntree, node, leaf

from sentence_dep_graph_view import spacy_display_sent_graph

from dependencies import  dep_node, make_sent_dep_tree, ANNOT_DEPTREE_KEY, SENTPRFX, x_breadth_level_recur

from anytree import RenderTree

ORIGKEY = 'origin'
TOKIDKEY = 'tok_id'
SENTIDKEY = 'sent_id'
TEXTKEY = 'text'
POSKEY =  'pos'        
MORPHKEY = 'morph'      
LEMMAKEY = 'lemma' 
STARTCHARKEY = 'start_char' 
ENDCHARKEY = 'end_char'   
HEADKEY = 'head'       
HEADTXTKEY = 'head_txt' 
DEPRELKEY = 'deprel'

#import pdb; pdb.set_trace()

TOKEN   = 0
UCSTART = 1
UCEND   = 2

#DEBUG = True
DEBUG = False

DEMO = True
#DEMO = False

# from: https://universaldependencies.org/u/dep/all.html#al-u-dep/dep
# "dep: unspecified dependency
# A dependency can be labeled as dep when it is impossible to determine a more precise relation.
# This may be because of a weird grammatical construction, or a limitation in conversion or parsing software.
# The use of dep should be avoided as much as possible."

# pap 20260422
# IGNORED_DEPS = [ 'dep' ]
IGNORED_DEPS = [  ]

SPACY_GN_SEED_SRC_RELS_EXCLUSION_LST     = [ 'compound', 'flat:name', 'name', 'goeswith' ]
SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST  = [ 'acl', 'acl:recl', 'acl:relcl', 'appos', 'appos', 'aux',
                                             'aux:pass', 'cc', 'cc:preconj', 'ccomp', 'conj', 'cop', 'csubj', 'csubjpass',
                                             'dep', 'discourse', 'dislocated', 'expl', 'expl:subj', 'expl:pass', 'expl:xcomp',
                                             'foreign', 'iobj',
                                             'list', 'mark', 'neg', 'nmod', 'nmod:tmod', 'nsubj', 'nsubjpass',
                                             'obj', 'obl', 'orphan', 'parataxis', 'punct', 'punct', 'reparandum',
                                             'root', 'sconj', 'vocative']
#SPACY_GN_SUBTREE_RELS_EXCLUSION_LST  = [ 'xcomp' ]
SPACY_GN_SUBTREE_TRGT_RELS_EXCLUSION_LST  = [ ]

NUMTYPES = [ int, float  ]

 
class document_spacyfier( object ):

     def __init__( self, lang = 'en' ):
          if lang == 'en':
               self.nlp = spacy.load( 'en_core_web_sm' )
          elif  lang == 'fr':
               self.nlp = spacy.load( 'fr_core_news_sm' )
          else:
               assert( 0 )
          # Lemmatization (V3.0)
          # spaCy provides two pipeline components for lemmatization:
          # 1. The Lemmatizer component provides lookup and rule-based lemmatization methods
          # in a conﬁgurable component. An individual language can extend the Lemmatizer 
          # as part of its language data.
          # 2. The EditTreeLemmatizer (V3.3)
          # component provides a trainable lemmatizer.     
          self.lemmatizer = self.nlp.get_pipe( 'lemmatizer' )
          self.lemmatizer_mmode = self.lemmatizer.mode    #  either 'rule' or 'EditTree'
          # doc = nlp("I was reading the paper.") ; print([token.lemma_ for token in doc])
          self.reset()

     def reset( self, doctext = '' ):
          assert( type( doctext ) is str )
          self.doctext = doctext
          self.parsed_doc = None
          self.chunks_spacy = set()
          self.curr_tok_idx = 0
          self.tokens = []
          self.curr_sent_idx = 0
          self.sents = []

     def tstart( self, tok ):
          assert( type( tok ) is spacy.tokens.token.Token )
          return tok.idx
          
     def tend( self, tok ):
          assert( type( tok ) is spacy.tokens.token.Token )
          return tok.idx + len( tok.text )

     def __repr__( self ):
          print( 'class document_spacyfier with text == {0}'.format( self.doctext ))

     def __str__( self ):
          return '{0}'.format( self.doctext ) 
       
     def depparse_doctext( self, doctext = '' ):
          self.reset( doctext )
          self.parsed_doc = self.nlp( doctext )  # Note parse_doc is a spacy class Doc instance not a list
          # sets up sentence list
          self.sents = list( self.parsed_doc.sents ) # Note self.parsed_doc.sents is an iterator not a list
          self.curr_sent_idx = 0
          # sets up token list
          self.curr_tok_idx = 0
          self.tokens = list( self.parsed_doc )
          ##          # -----
          if DEBUG:
               idx = 0
               for k in self.tokens:
                    assert( k.i == idx )
                    assert( k.sent.start <= k.i )
                    assert( k.i < k.sent.end )
                    print( 'token idx== {0} i== {1}  text== {2} sent_start_tok_id== {3} sent_end_tok_id== {4}'.format( idx, k.i, k.text,
                                                                                                                       k.sent.start, k.sent.end ))
                    idx += 1
          
          for chunk in self.parsed_doc.noun_chunks:
               tok_beg = self.parsed_doc[ chunk.start ]
               tok_end = self.parsed_doc[ chunk.end - 1   ]
               tok_beg_pos = tok_beg.idx
               tok_end_pos = tok_end.idx + len( tok_beg.text )
               self.chunks_spacy.add( ( doctext[ tok_beg_pos : tok_end_pos ], tok_beg_pos, tok_end_pos) )

          if DEBUG:
               # for viewing using an internet navigator
               answ = input( 'Display deps? y/o' )
               if answ == 'y':
                    displacy.serve( self.parsed_doc,  style = 'dep' )
               else:
                    pass
                    
     def next_tok( self ):
          # NOTE: my token idx and sentence idx are numbered from 0 (since they are python list indices)
          if self.curr_tok_idx >= len( self.tokens ) or self.curr_sent_idx >= len( self.sents ) :
               return None
          else:
               # we get the first not yet returned token in the current sentence
               tok = self.tokens[ self.curr_tok_idx ]
               assert( tok.i == self.curr_tok_idx ) 
               assert( self.sents[ self.curr_sent_idx ].start <= tok.i )
               if self.curr_tok_idx == self.sents[ self.curr_sent_idx ].end:
                    # we are pointing at the first token of the next sentence, i.e. during the last call to next() we have crossed a sentence boundary.
                    assert( (self.curr_sent_idx + 1) < len( self.sents ) ) # since we have a token it must belong to a sentence
                    self.curr_sent_idx += 1
                    assert( tok.i <= self.sents[ self.curr_sent_idx ].start )
               else:
                    assert( tok.i < self.sents[ self.curr_sent_idx ].end )
               annots = { ORIGKEY     : 'spacy',
                          TOKIDKEY         : self.curr_tok_idx,
                          SENTIDKEY    : self.curr_sent_idx,
                          TEXTKEY       : tok.text,
                          POSKEY        : tok.pos_, # coarse grain POS
                          MORPHKEY      : tok.morph, # fine grained POS Spacy statistical morphology
                          LEMMAKEY      : tok.lemma_,
                          STARTCHARKEY : tok.idx,
                          ENDCHARKEY   : tok.idx + len( tok.text ),
                          HEADKEY       : tok.head.i, # token idx of the head
                          HEADTXTKEY   : tok.head.text,
                          DEPRELKEY     : tok.dep_  }
               res = indexed_object( myrepr = annots[ TEXTKEY ], idx = self.curr_tok_idx, annotations = annots )
               # -----
               self.curr_tok_idx += 1
               return res
     
     def process_document( self, text ):
          assert( type( text ) is str )
          self.depparse_doctext( text )
          tmp_list_spacy = []
          indexed_tok_spacy = self.next_tok()
          while indexed_tok_spacy:
               if DEBUG:
                    print( 'in spacy process_document next token is i== {0} tok== {1} annots== {2}'.format( indexed_tok_spacy.idx,
                                                                                                            indexed_tok_spacy.myrepr,
                                                                                                            indexed_tok_spacy.annotations ) )
               tmp_list_spacy.append( indexed_tok_spacy )
               indexed_tok_spacy = self.next_tok()
          if DEBUG:
               print( 'type( tmp_list_spacy ) == {0} res== {1}'.format( type( tmp_list_spacy), tmp_list_spacy ))
               print( '' * 40 )
               for tk in tmp_list_spacy:
                   print( '{0} \t{1}'.format( tk, tk.annotations ))
                   print( '' * 40 )
          return tmp_list_spacy
     
def import_spacy_sentence_tokens_and_deps( i = 0, spacy_word_indexed_obj_lst = [], doc = None ):
     # note: i is the current spacy word idx of first token to process, words are numbered
     # from 0 to n in a increasing succesion throughout the whole document.
     curr_snt_mwus = {}
     tk = None
     while i < len( spacy_word_indexed_obj_lst ):
          tk = spacy_word_indexed_obj_lst[ i ]
          curr_spacy_sid = tk.annotations[ SENTIDKEY ]
          new_sid = SENTPRFX + str( curr_spacy_sid )
          while (tk.annotations[ SENTIDKEY ] == curr_spacy_sid ):
               curr_sent_tok_idx  = len( curr_snt_mwus.keys() ) # we take the len() and not len()-1 because we have not yet appended the mwu to the list.
               if DEBUG:
                    print( 'DEBUG creating new mwu sid== {0} tkid== {1} nm== {2}'.format( str(tk.annotations[ SENTIDKEY ]),
                                                                                          str(tk.annotations[ TOKIDKEY ]),
                                                                                          (SENTPRFX + str(tk.annotations[ SENTIDKEY ]) +
                                                                                           mwu.default_mwu_name_prefix +
                                                                                           str(tk.annotations[ TOKIDKEY ]))))
                                                                                     
               u = mwu( nm = new_sid + mwu.default_mwu_name_prefix + str( tk.annotations[ TOKIDKEY ]),
                        txtsps = [ (tk.annotations[ STARTCHARKEY ], tk.annotations[ ENDCHARKEY ]) ],
                        typ = 'spacy_word',
                        annotations = tk.annotations )
               curr_snt_mwus[ i ] = u
               u.annotations[ 'spacy_sent_tok_idx' ] = curr_sent_tok_idx
               doc.add_mwu( u )
               i += 1
               if i >= len( spacy_word_indexed_obj_lst ):
                    break
               else:
                    curr_spacy_sid = tk.annotations[ SENTIDKEY ]
                    tk = spacy_word_indexed_obj_lst[ i ]
          # NOTE: from here to the end of the function i should NOT be modified since
          # it points to the first token of the next sentence or to a position nust after
          # the end of the file. Same thing for tk.
          
          assert( new_sid not in doc.constructions[ 'sentences' ].keys() )
          sent_construct = construction( nm = new_sid,
                                         typ = 'spacy_sent',
                                         mwus = { name( m ) : m for m in curr_snt_mwus.values()} )
          doc.constructions[ 'sentences' ][ name( sent_construct ) ] = sent_construct

          if DEBUG:
               print('======= sent_construct.mwus.keys()== {0}'.format( sent_construct.mwus.keys() ))
               for m in sent_construct.mwus.values():
                    print( '\ttype(sent_construct.mwu) is {0} sent_construct.mwu {1}'.format( type( m),   m ))
          
          spacy_deps_to_relations( new_sid, curr_snt_mwus, name( sent_construct ), doc )
          return (i, name( sent_construct )) 
               
def spacy_deps_to_relations( new_sid, snt_mwus, snt_contruct_nm, doc ):
     curr_snt_rels = {}
     r_num  = 0
     if len( snt_mwus ) > 0:
          for tk in snt_mwus.values():
              assert( tk.name.index( new_sid ) == 0 )
              if tk.annotations[ DEPRELKEY ] == 'ROOT':
                   rel = relation( nm  = new_sid + 'R{0}'.format( r_num ),
                                   src = snt_mwus[ tk.annotations[ TOKIDKEY ] ], 
                                   trg = snt_mwus[ tk.annotations[ TOKIDKEY ] ],
                                   annotations = { SENTIDKEY : tk.annotations[ SENTIDKEY ]},
                                   typ = 'ROOT' )
              elif tk.annotations[ DEPRELKEY ] in IGNORED_DEPS:
                   sys.stderr.write( 'WARNING ignoring dependency of type {0} with source {1}\n'.format( tk.annotations[ DEPRELKEY ],  name( tk ) ) )
                   # NOTE: this type of dependendy of little interest except in very rare cases is
                   # ignored here because in most of the cases it results from the spacy parsing of inter-sential
                   # spaces or line-breaks of all kinds.
              else:
                   rel = relation( nm  = new_sid + 'R{0}'.format( r_num ),
                                   src = snt_mwus[ tk.annotations[ HEADKEY ] ], 
                                   trg = snt_mwus[ tk.annotations[ TOKIDKEY ] ],
                                   annotations = { SENTIDKEY : tk.annotations[ SENTIDKEY ]},
                                   typ = tk.annotations[ DEPRELKEY ] )
              r_num += 1
              curr_snt_rels[ name( rel )  ] = rel 
          for r in curr_snt_rels.values():
               doc.add_rel( r )
          doc.constructions[ 'sentences' ][ snt_contruct_nm ].rels  = curr_snt_rels
          #doc.rels.update( curr_snt_rels )
     else:
          pass

def escape_metachar( s ):
     return s.replace( '\\', '\\\\').replace( '\n', '\\012').replace( '\t', '\\011').replace( "'", "\\'" )

def spacy_mwu_to_csv_header_col_lst( m ):
     annot_keys = list( m.annotations.keys() )
     if ANNOT_DEPTREE_KEY in annot_keys:
          annot_keys.remove( ANNOT_DEPTREE_KEY )
     else:
          pass
     return annot_keys

def spacy_mwu_to_csv_str( m ):
     out_msg = ''
     annot_keys = list( m.annotations.keys() )
     if ANNOT_DEPTREE_KEY in annot_keys:
          annot_keys.remove( ANNOT_DEPTREE_KEY )
     else:
          pass
     for k in annot_keys:
          if k == TEXTKEY:
             out_msg += '{0}\t'.format( escape_metachar( m.annotations[ k ] ))
          else:
               if type( m.annotations[ k ] ) in NUMTYPES:
                    out_msg += '{0}\t'.format( m.annotations[ k ] )
               else:
                    out_msg += '{0}\t'.format( escape_metachar( '{0}'.format( m.annotations[ k ])))
     return out_msg

def spacy_export_doc_mwus_to_csv( doc, out_strm ):
        # export all existing mwus from the current doc into out_strm  in csv format
        # M/R is a class label for the current entry M=mwu and R=relation
        assert( len( list( doc.mwus.keys() )) > 0 )
        
        header = '{0}\t'.format( '#docid' )
        header += '\t'.join( spacy_mwu_to_csv_header_col_lst( list( doc.mwus.values() )[ 0 ] ))
        out_strm.write( '{0}\n'.format( header ))
        
        for m_nm in doc.mwus.keys():
            m = doc.mwus[ m_nm ]
            out_strm.write( '{0}\t'.format( doc.name ))
            out_strm.write( '{0}\n'.format( spacy_mwu_to_csv_str( m )))

def spacy_export_doc_rels_to_csv( doc, out_strm ):
        # export all existing relations from the current doc into out_strm  in csv format
        # M/R is a class label for the current entry M=mwu and R=relation
        header = '{0}\t{1}\t{2}\t{3}\t'.format('#docid', ORIGKEY, 'rel_id', 'rel_type')
        header += '\t'.join( list( map( lambda  x :  'src_'  + x, spacy_mwu_to_csv_header_col_lst( list( doc.mwus.values() )[ 0 ] ))))  
        header += '\t'
        header += '\t'.join( list( map( lambda  x :  'trgt_' + x, spacy_mwu_to_csv_header_col_lst( list( doc.mwus.values() )[ 0 ] ))))
        out_strm.write( '{0}\n'.format( header ))
        for r_nm in doc.rels.keys():
           r = doc.rels[ r_nm ]
           if DEBUG:
                pprint( 'exporting relation {0} to csv'.format( r_nm ) )
           out_strm.write( '{0}\t{1}\t{2}\t{3}\t'.format( doc.name, 'Spacy', r.name, r.typ ))
           out_strm.write( spacy_mwu_to_csv_str( r.src  ) )
           out_strm.write( spacy_mwu_to_csv_str( r.trgt ) )
           out_strm.write( '\n' )
        out_strm.flush()

def export_mwu_lst( doc, mwus, out_strm ):
     mwus.sort( reverse=False, key=lambda x : (x.txtspans[0][0], x.txtspans[0][1]) )
     msg = ''
     for m in mwus:
          msg += (' ' + (m.text( doc ).replace('\t', ' ')))
     msg = msg.strip()
     out_strm.write( msg + '\n' )


def match_dep( rel, typ_pred = (lambda x: False), src_pred = (lambda x: False), trgt_pred = (lambda x: False),
               annot_pred = (lambda x: False), result_location = 'src' ):
     # return either None, i.e. no match or either the src
     # or the trgt of rel depending on the value of result location
     # which can be either 'src' or 'trgt'. 

     assert( type( rel  ) is relation )
     assert( result_location  in [ 'src', 'trgt' ] )

     match_p = (typ_pred( rel.typ) and
                src_pred( rel.src ) and
                trgt_pred( rel.trgt ) and
                annot_pred( rel.annotations ) )
     
     if match_p ==  True:
          if result_location == 'src':
               return rel.src
          else:
               assert( result_location == 'trgt')
               return rel.trgt
     else:
          return None
     

def grow_dep_subtrees( doc, snt, seed_core_noun_mwu_lst, max_depth = 100 ):
     # extract the dependency subtree associated to each core noun mwu and 
     # create one construction for each noun group seed word
     global SPACY_GN_SEED_SRC_RELS_EXCLUSION_LST
     global SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST
     global SPACY_GN_SUBTREE_TRGT_RELS_EXCLUSION_LST

     def subtree_src_typ_pred( typ ):
          return typ not in SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST
     def subtree_trgt_typ_pred( typ ):
          return typ not in SPACY_GN_SUBTREE_TRGT_RELS_EXCLUSION_LST

     noun_group_res_lst = []  # one pattern is created associated to each noun token found previously
     m_kstruct = None
     outer_border= []
     for m in seed_core_noun_mwu_lst:
          if DEBUG:
               print( '>>>for loop0 in grow_dep_subtree m {0}'.format( m ))
          m_kstruct = construction( nm = 'noun_subtree_' + str( name( m )), typ = 'coreNG', mwus = {}, rels = {}, opt_mwus = {}, opt_rels = {} )
          outer_border += [ m ]
          if m not in m_kstruct.mwus:
               m_kstruct.add_mwu( m )
          else:
               pass
          noun_group_res_lst.append( m_kstruct )

     k = 0
     assert( len( outer_border ) != 0 )
     for m in seed_core_noun_mwu_lst:
          # each core noun will be the root of a dependency subtree that we grow below.
          outer_border = [ m ]
          new_outer_border = []
          while outer_border:
               for m_out in outer_border:
                    for r in snt.rels.values():
                         if match_dep( rel = r, typ_pred   = subtree_src_typ_pred, src_pred = (lambda   src: True),
                                       trgt_pred  = (lambda  trgt: trgt == m_out), annot_pred = (lambda annot: True),
                                       result_location = 'src'):
                              if r.src not in noun_group_res_lst[ k ].mwus:
                                   if DEBUG:
                                        print( '>>>ADDING C rel: {0} with construction {1}'.format( r, str( name( m_kstruct )) ))
                                   noun_group_res_lst[ k ].add_mwu( r.src )
                                   assert( r.src  not in new_outer_border )
                                   new_outer_border.append( r.src )
                                   assert( r not in  noun_group_res_lst[ k ].rels  )
                                   noun_group_res_lst[ k ].add_rel( r )
                              else:
                                  pass     
                         else:
                              pass
               outer_border = new_outer_border  # pop the outer_border old level and append the new level
               new_outer_border = []
          k += 1    
            
     if DEBUG:
          print( 'DEBUG2 noun_group_res_lst is=={0}'.format( noun_group_res_lst ))
          for c in noun_group_res_lst:
               print( c )
               for m in c.mwus:
                    print( doc.mwus[ m ] )
               for r in c.rels:
                    print( doc.rels[ r ] )
          print( '-----------------')
     return noun_group_res_lst


def spacy_pattern_filter_noun_group( doc, exclude_determiner_p = True ):
     global SPACY_GN_SEED_SRC_RELS_EXCLUSION_LST
     global SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST

     if exclude_determiner_p:
        SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST += [ 'det', 'det:predet' ]
     else:
         pass
     for snt in doc.constructions[ 'sentences' ].values():
          for r_seed in SPACY_GN_SEED_SRC_RELS_EXCLUSION_LST:
               assert( r_seed not in SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST)
               # we want to consider in the nominal subtree the following dependencies:
               # ---------relations where TARGET is either the core noun or an outer border word
               # compound, compound:prt, flat, goeswith amod, advmod (when
               # modifying an adjective), conj (when the source and the head
               # are adjectives or adverbs), ------------------------- NOTE:
               # This cases below are not handled for now
               # ------------------------  'nmod:poss' only when the SOURCE
               # is a possessive determiner --------- relations where the
               # SOURCE is either the core noun or an outter border word #
               # 'nmod:npmod' only when SOURCE is a noun (exclude pronouns) #
               # NOTE: stanza annotates nmod:npmod as obl:npmod  ; what about
               # Spacy ? 'nummod' # 'name' relation seems to exist in udped,
               # is it produced by stanza ? by Spacy ?
               #-----------
    
          if DEBUG:
               print( 'processing doc== {0}'.format( doc.name ))
        
          # ---- step0 finding all noun words
          # select all words with upos NOUN, PROPN 
          core_noun_mwu_lst = []
          for  m in snt.mwus.values():
               if DEBUG:
                    print( 'TESTING m{0}'.format( m  )) 
               if m.typ == 'spacy_word':
                    if  m.annotations[ POSKEY ] in [ 'NOUN', 'PROPN' ]:
                         core_noun_mwu_lst.append( m )
                    else:
                         pass
               else:
                    pass
               if DEBUG:
                    print( 'spacy_pattern_filter_noun_group>>>>>> DEBUG++++++++++++++++')
                    print( '\t\t'.format( m.typ, m.name ))
                    print( '\t\tmwu.name== {0} m.typ== {1} m.annotations== {2}\t\t\nnoun_mwu_lst=={3}'.format( m.name, m.typ,
                                                                                                               [ i for i in m.annotations.items() ],
                                                                                                               core_noun_mwu_lst ))
                    print( '++++++++++')
          if DEBUG:
               print( 'BEFORE expanding SUBTREES  noun word list ==  {0}'.format( [ x.text( doc ) for x in core_noun_mwu_lst ] ))
               print( 'len( core_noun_mwu_lst) {0} core_noun_mwu_lst {1}'.format( len(core_noun_mwu_lst), core_noun_mwu_lst ))

          return grow_dep_subtrees( doc, snt, core_noun_mwu_lst )
     
def spacy_filter_noun_groups( doc, out_strm, exclude_determiner_p = True ):
        # subject verb object / attribute and their modifiers extraction
        # kstruct_lst = pattern_filter_svo_O_ADJ_triplet( list( doc.rels.values() ) )
        
        doc.constructions[ 'noun_groups' ] = spacy_pattern_filter_noun_group( doc, exclude_determiner_p )
        
        assert( type( doc.constructions[ 'noun_groups' ] ) is list )
        assert( False not in [ (type( x ) is construction) for x in doc.constructions[ 'noun_groups' ] ] )

        for k in doc.constructions[ 'noun_groups' ]:
             if DEBUG:
                msg = '++++++++++++++++++++++++++ noun groups filtering result for '
                msg += 'file {0} has found mwus {1} match(es) and rels {2} match(es): '.format( doc.name, len( k.mwus.keys()), len( k.rels.keys() ) )
                msg += ' in construction {0} of typ== {1}\n'.format( k.name, k.typ )
                print( msg )
                print_mwus( k.mwus, doc )
                print_rels( k.rels )
                print( '===========' )
             export_mwu_lst( doc, list(k.mwus.values()), out_strm ) 

ud_fr_rel_samples = { 'compound'      : {'txt': 'ambiance chalet',  'deps': [{'src':'ambiance', 'trgt':'chalet'}]},
                      'name'          : {'txt': 'Cervelo test team', 'deps': [{'src':'team', 'trgt':'Cervelo'}, {'src':'team', 'trgt':'test'}]},
                      'flat:name'     : {'txt': 'Leur présidente est Shazza Nzingha', 'deps':[{'src':'Shazza', 'trgt':'Nzingha'}]}, 
                      'flat:foreign'  : {'txt': "C'est le créateur de The Tenth", 'deps':[{'src':'The', 'trgt':'Tenth'}]},
                      'goeswith'      : {'txt': 'Nous avons testé le restaurant ce week end', 'deps':  [{'src':'week', 'trgt':'end'}]}
                      }

# examples from SpaCy source code.
spacy_sentences = [
    "Apple cherche à acheter une start-up anglaise pour 1 milliard de dollars",
    "Les voitures autonomes déplacent la responsabilité de l'assurance vers les constructeurs",
    "San Francisco envisage d'interdire les robots coursiers sur les trottoirs",
    "Londres est une grande ville du Royaume-Uni",
    "L’Italie choisit ArcelorMittal pour reprendre la plus grande aciérie d’Europe",
    "Apple lance HomePod parce qu'il se sent menacé par l'Echo d'Amazon",
    "La France ne devrait pas manquer d'électricité cet été, même en cas de canicule",
    "Nouvelles attaques de Trump contre le maire de Londres",
    "Où es-tu ?",
    "Qui est le président de la France ?",
    "Où est la capitale des États-Unis ?",
    "Quand est né Barack Obama ?",
    "On est alors amené à définir pour les besoins de l’évaluation un formalisme intermédiaire suffisamment spécifique pour que l’évaluation soit pertinente, mais assez général pour permettre aux différentes approches de se comparer sur un terrain commun. Souvent ce sont les participants eux-mêmes qui définissent la projection des sorties de leur système vers le formalisme commun, car ils en ont une meilleure connaissance que quiconque d’autre. Certains ont proposé de généraliser cette mise en correspondance en générant de manière systématique tous les appariements possibles (Teufel, 1995)."
]

def spacy_dep_ng_node_root_p( a_node ):
     assert( len( a_node.mwus.keys()) == 1 )
     the_mwu = (list( a_node.mwus.values()))[ 0 ]
     return ((the_mwu.annotations[ POSKEY ] in [ 'NOUN', 'PROPN' ]) and
             (the_mwu.annotations[ DEPRELKEY ] not in SPACY_GN_SEED_SRC_RELS_EXCLUSION_LST ))

def spacy_dep_ng_node_subtree_p( a_node ):
     assert( len( a_node.mwus.keys()) == 1 )
     the_mwu = (list(a_node.mwus.values()))[ 0 ]
     return (the_mwu.annotations[ DEPRELKEY ] not in SPACY_GN_SUBTREE_SRC_RELS_EXCLUSION_LST)

def dep_tree_level_filter_NG( a_node_lst, nlevel, doc ):
     # function argument of x_breadth_level_recur() in dependencies.py
     print( 'DEBUGlast dep_tree_level_filter_NG '  )  
     for nd in a_node_lst:
          print( '\t_\t_\tDEBUG level== {0} nd== {1} spacy_dep_ng_node_root_p( nd )== {2} text== {3}'.format( nlevel,  nd, spacy_dep_ng_node_root_p( nd ),
                                                                                                              list(nd.mwus.values())[0].text( doc  )))
##     return [ nd for nd in a_node_lst if spacy_dep_ng_node_root_p( nd )  ]
     res =[ nd for nd in a_node_lst if ((list(nd.mwus.values())[0].annotations[ POSKEY ]) in [ 'NOUN', 'PROPN' ]) ]
     print( 'dep_tree_level_filter_NG() ', res )
     return res

def output_anytree_dep( outfdesc = None, dptree = None, snt = None, doc = None ):
     print( 'DEBUG output_anytree_dep_subtree dptree= ', name( dptree ), ' children= ', [ name(x) for x in dptree.children ] )
     assert( type( outfdesc ) is not None )
     assert( type( dptree  ) is dep_node )
     assert( type( snt ) is construction )
     assert( type( doc ) is document )

     for pre, fill, node in RenderTree( dptree  ):
          nd_mwu = node.mwus[ list( node.mwus.keys())[ 0 ] ]
          tree_str = u"%s%s" % (pre, name( nd_mwu ) + '/' + nd_mwu.text( doc ) + '/' +
                                nd_mwu.annotations[ POSKEY ] + '/' + nd_mwu.annotations[ DEPRELKEY ])
          outfdesc.write( '{0}\n'.format( tree_str ))
          if DEBUG:
               print( 'DEBUG >>> {0}'.format( node.mwus[ list( node.mwus.keys())[ 0 ] ], )) 

def demo_spacy_fr( ):
     global DEBUG
     parser =  document_spacyfier( lang = 'en'  )
     some_text =  "He has been playing for several hours."

     indexed_parsed_text = parser.process_document( some_text  )
    
     curr_spacy_word_idx = 0
     
     doc = document( nm = 'demo_spacy_fr', content = some_text, metadata = 'pap 20260608' )

     doc.constructions[ 'sentences' ] = {}
     res_ng_lst = []
     while curr_spacy_word_idx < len( indexed_parsed_text ):
          spacy_sent_first_tok_offset = curr_spacy_word_idx
          ( curr_spacy_word_idx, curr_sid ) = import_spacy_sentence_tokens_and_deps( i = curr_spacy_word_idx,
                                                              spacy_word_indexed_obj_lst = indexed_parsed_text,
                                                              doc = doc )
     # if here, we first import all sentences of a document, then process each one in sequence,
     # drawback we load the whole document in memory but we can process inter-sentence relations.
     # for s in doc.constructions[ 'sentences' ].keys():
          if curr_spacy_word_idx  <= len( indexed_parsed_text):
               snt = doc.constructions[ 'sentences' ][ curr_sid ]
               print('-' * 100 )
               print( curr_spacy_word_idx, ' ', curr_sid )
               if DEMO:
                    spacy_display_sent_graph( snt, doc, show_graph = True)

                    dpt = make_sent_dep_tree( snt, doc )

                    with open( './' + name( doc ) + '_' + curr_sid + '_anytree_dep.txt', 'w') as out_anytree:
                         output_anytree_dep( outfdesc = out_anytree, dptree = dpt, snt = snt, doc = doc )
                         
                    with open( './' + name( doc ) + '_' + curr_sid + '_anytree.txt', 'w' ) as anytree_out:
                         snt_txt  = ' '.join( [  getobj( m_nm ).text( doc ) for m_nm in snt.mwus ] )
                         anytree_out.write( '# {0}\n'.format( snt_txt ))
                         
                         for pre, fill, node in RenderTree( dpt  ):
                              tree_str = u"%s%s" % (pre, name( node ))
                              anytree_out.write( '{0}\n'.format( tree_str ))
                              if DEBUG:
                                   print( 'DEBUG >>> {0}'.format( node.mwus[ list( node.mwus.keys())[ 0 ] ]))

                    res  =  x_breadth_level_recur( [ dpt ], doc = doc, level_node_lst_proc_fun = dep_tree_level_filter_NG,
                                                   ng_root_p  = spacy_dep_ng_node_root_p )
                    aux_frmt ='\nRESULT*****>> curr_sid == {0} res x_breadth_left_node == {1}\n'
                    print( aux_frmt.format( curr_sid, res  ))

                    with open( './' + name( doc ) + '_' + curr_sid + '_anytree_dep_NG.txt', 'w') as out_anytree:
                         for ng_node in res:
                              print( 'ng_node is ', name( ng_node ))
                              output_anytree_dep( outfdesc = out_anytree, dptree = ng_node, snt = snt, doc = doc )
               else:
                    pass

               demo_spacy_noun_groups_out = './demo_spacy_noun_'+ name( snt ) +'_groups.csv'

          else:
               pass

     demo_spacy_mwus_out = './demo_spacy_' + name( doc ) + '_mwus.csv'
     with open( demo_spacy_mwus_out, 'w' ) as mwus_out:
          spacy_export_doc_mwus_to_csv( doc, mwus_out )
                    
     demo_spacy_rels_out = './demo_spacy_' + name( doc ) + '_rels.csv'
     with open( demo_spacy_rels_out, 'w' ) as rels_out:
          spacy_export_doc_rels_to_csv( doc, rels_out )

     return( doc  )

if DEMO:
     d = demo_spacy_fr()
else:
     pass
