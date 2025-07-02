# GAP script to define Ih as a permutation group on 12 vertices and export permutations

# Define Ih as a permutation group on 12 points
a := (1,2,3);
b := (1,2)(3,4);
inv := (1,12);
Ih := Group(a, b, inv);

# Get all elements (actual permutations)
elements := Elements(Ih);

Print(elements);

# Write all permutations as image lists to a file
PrintTo("ih_permutations.txt", "");

for g in elements do
  Print(g, " — IsPerm: ", IsPerm(g), "\n");
od;

stream := OutputTextFile("ih_permutations.txt", false);
for g in elements do
  perm_list := List([1..12], i -> i^g);
  AppendTo(stream, perm_list, "\n");
od;
CloseStream(stream);

#for i in [1..Length(elements)] do
 #   # Build image list explicitly using ListPerm
  #  perm := ListPerm([1..12], elements[i]);   # returns image list
   # perm := List(perm, x -> x - 1);           # convert to 0-based for Python
    #AppendTo("ih_permutations.txt", perm, "\n");
#od;