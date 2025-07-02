
# ih_irreps.g — compute irreps of the icosahedral group Ih

# Define the alternating group A5 as a permutation group on 12 points
# a := (1,2,3);
# b := (1,2)(3,4);  # These two generate A5
# I := Group(a, b);
I := AlternatingGroup(5);

# Add inversion (a transposition not in A5) to get Ih
# inv := (1,12);    # Inversion operation
# Ih := Group(a, b, inv);
Ih := DirectProduct(I, CyclicGroup(2));

# Print group size (should be 120)
Print("Order of Ih: ", Size(Ih), "\n");

# Compute and display character table
tbl := CharacterTable(Ih);
Display(tbl);

# Save irreducible characters to files
irrs := Irr(tbl);
for i in [1..Length(irrs)] do
    fname := Concatenation("ih_irrep_", String(i), ".txt");
    PrintTo(fname, irrs[i], "\n");
od;