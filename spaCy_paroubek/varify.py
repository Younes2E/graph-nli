from copy import deepcopy

def varify(  obj, name_slot_label = 'name'):
      print( 'assert( type( obj.{0}) is str )'.format( name_slot_label  ))
      exec( 'assert( type( obj.{0}) is str )'.format( name_slot_label  ))
      print( 'assert( obj.{0} != "" )'.format( name_slot_label  ))
      exec( 'assert( obj.{0} != "" )'.format( name_slot_label  ))
      print( 'assert( obj.{0}[ 0 ].isalpha() and obj.{0}[ 1: ].isalnum() )'.format( name_slot_label  ))
      exec( 'assert( obj.{0}[ 0 ].isalpha() and obj.{0}[ 1: ].isalnum() )'.format( name_slot_label  ))
      s1 = eval( 'deepcopy( obj.' + name_slot_label + ')' )
      print( '>>>s1', s1 )
      s2 = s1 + ' = obj'
      print( '>>>s2== ', s2 )
      exec( s2 )
      print( str(eval(s1)) )
      return 
      
# exec(str( [c for c in 'word0 = m'] )))
