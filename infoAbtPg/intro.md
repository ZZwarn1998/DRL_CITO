# Introduction
## Meaning of files
&emsp;There are five files in each folder.
- `Attr_Method_deps.csv` contains the information about method and attribute dependence;
- `CId_importance.csv` contains the information about `importance of class` which is obtained by HITS algorithm;
- `CId_Name.csv contains` the information about class ID and the name of class;
- `Couple_List.csv` contains the information about coupling value between classes;
- `deps_type.csv` contains the information about dependent relationship between classes.  

## Types of relationships

There are five types of relationships between classes.

- **NONE = 0** , no relationship
- **As = 1**,  dependency|simple aggregation|association
- **Ag = 2**,  composition| association constrained by strict lifecycle
- **I = 3**,  generalization (or inherit)
- **Dy = 4**,  dynamic dependency