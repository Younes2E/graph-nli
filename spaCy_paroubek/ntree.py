class ntree ():
  def __init__( self, nm = 'a_binary_tree' ):
       self.root = None
       self.label = nm

  def __repr__( self ):
      return( 'ntree {0}'.format( self.label ))

  def show( self ):
      res = self.label + '='
      if self.root is not None:
          res += self.root.show()
      else:
          res += 'empty'
      return res

  def set_root( self, r=None ):
       self.root = r

class node ():
   def __init__( self, nm = 'a_node' ):
       self.label = nm
       self.children = []

   def show( self ):
       res = '( ' + self.label + ' ( '
       for c in self.children:
         res += c.show()
       res += ' ) )'
       return res

   def set_left( self, lnd):
       self.left = lnd

   def add_child( self, nd ):
       if nd not in self.children:
         self.children.append( nd )
       else:
         print( 'Warning node {0} already exists, it will not be added.')
         pass

class leaf():
   def __init__( self, nm = 'a_leaf' ):
       self.label = nm

   def show( self ):
       return self.label
