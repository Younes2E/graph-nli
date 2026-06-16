from anytree import Node, RenderTree

x = Node( name = 'demo0', width=3, length=9 )
print( type( x.root ))
print( x.root )
print( RenderTree( x.root ))

udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jet", parent=dan)
jan = Node("Jan", parent=dan)
joe = Node("Joe", parent=dan)

print( type( udo.root ))
print( udo.root )

for pre, fill, node in RenderTree( udo ):
  treestr = u"%s%s" % (pre, node.name)
  print( treestr )
