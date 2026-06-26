import sys
import spacy
from spacy import displacy
from .text_span_v2 import mwu, relation, construction, name



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
SENTPRFX = 'S'

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

NUMTYPES = [ int, float  ]

 

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

