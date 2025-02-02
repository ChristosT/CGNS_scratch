#include "petscsf.h"

#include <petscdmplex.h>
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/

#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkXMLUnstructuredGridWriter.h>

#if !defined(CGNS_ENUMT)
  #define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
  #define CGNS_ENUMV(a) a
#endif
// Permute plex closure ordering to CGNS
static PetscErrorCode DMPlexCGNSGetPermutation_Internal(DMPolytopeType cell_type, PetscInt closure_size, const int **perm)
{
  // https://cgns.github.io/CGNS_docs_current/sids/conv.html#unstructgrid
  static const int bar_2[2]   = {0, 1};
  static const int bar_3[3]   = {1, 2, 0};
  static const int bar_4[4]   = {2, 3, 0, 1};
  static const int bar_5[5]   = {3, 4, 0, 1, 2};
  static const int tri_3[3]   = {0, 1, 2};
  static const int tri_6[6]   = {3, 4, 5, 0, 1, 2};
  static const int tri_10[10] = {7, 8, 9, 1, 2, 3, 4, 5, 6, 0};
  static const int quad_4[4]  = {0, 1, 2, 3};
  static const int quad_9[9]  = {
    5, 6, 7, 8, // vertices
    1, 2, 3, 4, // edges
    0,          // center
  };
  static const int quad_16[] = {
    12, 13, 14, 15,               // vertices
    4,  5,  6,  7,  8, 9, 10, 11, // edges
    0,  1,  3,  2,                // centers
  };
  static const int quad_25[] = {
    21, 22, 23, 24,                                 // vertices
    9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, // edges
    0,  1,  2,  5,  8,  7,  6,  3,  4,              // centers
  };
  static const int tetra_4[4]   = {0, 2, 1, 3};
  static const int tetra_10[10] = {6, 8, 7, 9, 2, 1, 0, 3, 5, 4};
  static const int tetra_20[20] = {
    16, 18, 17, 19,         // vertices
    9,  8,  7,  6,  5,  4,  // bottom edges
    10, 11, 14, 15, 13, 12, // side edges
    0,  2,  3,  1,          // faces
  };
  static const int hexa_8[8]   = {0, 3, 2, 1, 4, 5, 6, 7};
  static const int hexa_27[27] = {
    19, 22, 21, 20, 23, 24, 25, 26, // vertices
    10, 9,  8,  7,                  // bottom edges
    16, 15, 18, 17,                 // mid edges
    11, 12, 13, 14,                 // top edges
    1,  3,  5,  4,  6,  2,          // faces
    0,                              // center
  };
  static const int hexa_64[64] = {
    // debug with $PETSC_ARCH/tests/dm/impls/plex/tests/ex49 -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -dm_coord_petscspace_degree 3
    56, 59, 58, 57, 60, 61, 62, 63, // vertices
    39, 38, 37, 36, 35, 34, 33, 32, // bottom edges
    51, 50, 48, 49, 52, 53, 55, 54, // mid edges; Paraview needs edge 21-22 swapped with 23-24
    40, 41, 42, 43, 44, 45, 46, 47, // top edges
    8,  10, 11, 9,                  // z-minus face
    16, 17, 19, 18,                 // y-minus face
    24, 25, 27, 26,                 // x-plus face
    20, 21, 23, 22,                 // y-plus face
    30, 28, 29, 31,                 // x-minus face
    12, 13, 15, 14,                 // z-plus face
    0,  1,  3,  2,  4,  5,  7,  6,  // center
  };

  PetscFunctionBegin;
  *perm            = NULL;
  switch (cell_type) {
  case DM_POLYTOPE_SEGMENT:
    switch (closure_size) {
    case 2:
      *perm            = bar_2;
    case 3:
      *perm            = bar_3;
    case 4:
      *perm            = bar_4;
      break;
    case 5:
      *perm            = bar_5;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_TRIANGLE:
    switch (closure_size) {
    case 3:
      *perm            = tri_3;
      break;
    case 6:
      *perm            = tri_6;
      break;
    case 10:
      *perm            = tri_10;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    switch (closure_size) {
    case 4:
      *perm            = quad_4;
      break;
    case 9:
      *perm            = quad_9;
      break;
    case 16:
      *perm            = quad_16;
      break;
    case 25:
      *perm            = quad_25;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    switch (closure_size) {
    case 4:
      *perm            = tetra_4;
      break;
    case 10:
      *perm            = tetra_10;
      break;
    case 20:
      *perm            = tetra_20;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    switch (closure_size) {
    case 8:
      *perm            = hexa_8;
      break;
    case 27:
      *perm            = hexa_27;
      break;
    case 64:
      *perm            = hexa_64;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// typedef struct {
//   char infile[PETSC_MAX_PATH_LEN];  /* Input mesh filename */
//   char outfile[PETSC_MAX_PATH_LEN]; /* Dump/reload mesh filename */
// } AppCtx;

// static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
// {
//   PetscFunctionBeginUser;
//   options->infile[0]  = '\0';
//   options->outfile[0] = '\0';
//   PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
//   PetscCall(PetscOptionsString("-infile", "The input CGNS file", EX, options->infile, options->infile, sizeof(options->infile), NULL));
//   PetscCall(PetscOptionsString("-outfile", "The output CGNS file", EX, options->outfile, options->outfile, sizeof(options->outfile), NULL));
//   PetscOptionsEnd();
//   PetscCheck(options->infile[0], comm, PETSC_ERR_USER_INPUT, "-infile needs to be specified");
//   PetscCheck(options->outfile[0], comm, PETSC_ERR_USER_INPUT, "-outfile needs to be specified");
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

// // @brief Create DM from CGNS file and setup PetscFE to VecLoad solution from that file
// PetscErrorCode ReadCGNSDM(MPI_Comm comm, const char filename[], DM *dm)
// {
//   PetscInt degree;

//   PetscFunctionBeginUser;
//   PetscCall(DMPlexCreateFromFile(comm, filename, "ex16_plex", PETSC_TRUE, dm));
//   PetscCall(DMSetFromOptions(*dm));
//   PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

//   { // Get degree of the natural section
//     PetscFE        fe_natural;
//     PetscDualSpace dual_space_natural;

//     PetscCall(DMGetField(*dm, 0, NULL, (PetscObject *)&fe_natural));
//     PetscCall(PetscFEGetDualSpace(fe_natural, &dual_space_natural));
//     PetscCall(PetscDualSpaceGetOrder(dual_space_natural, &degree));
//     PetscCall(DMClearFields(*dm));
//     PetscCall(DMSetLocalSection(*dm, NULL));
//   }

//   { // Setup fe to load in the initial condition data
//     PetscFE        fe;
//     PetscInt       dim, cStart, cEnd;
//     PetscInt       ctInt, mincti, maxcti;
//     DMPolytopeType dm_polytope, cti;

//     PetscCall(DMGetDimension(*dm, &dim));
//     // Limiting to single topology in this simple example
//     PetscCall(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
//     PetscCall(DMPlexGetCellType(*dm, cStart, &dm_polytope));
//     for (PetscInt i = cStart + 1; i < cEnd; i++) {
//       PetscCall(DMPlexGetCellType(*dm, i, &cti));
//       PetscCheck(cti == dm_polytope, comm, PETSC_ERR_RETURN, "Multi-topology not yet supported in this example!");
//     }
//     ctInt = cti;
//     PetscCallMPI(MPIU_Allreduce(&ctInt, &maxcti, 1, MPIU_INT, MPI_MAX, comm));
//     PetscCallMPI(MPIU_Allreduce(&ctInt, &mincti, 1, MPIU_INT, MPI_MIN, comm));
//     PetscCheck(mincti == maxcti, comm, PETSC_ERR_RETURN, "Multi-topology not yet supported in this example!");
//     PetscCall(PetscPrintf(comm, "Mesh confirmed to be single topology %s\n", DMPolytopeTypes[cti]));
//     PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 5, dm_polytope, degree, PETSC_DETERMINE, &fe));
//     PetscCall(PetscObjectSetName((PetscObject)fe, "FE for VecLoad"));
//     PetscCall(DMAddField(*dm, NULL, (PetscObject)fe));
//     PetscCall(DMCreateDS(*dm));
//     PetscCall(PetscFEDestroy(&fe));
//   }

//   // Set section component names, used when writing out CGNS files
//   PetscSection section;
//   PetscCall(DMGetLocalSection(*dm, &section));
//   PetscCall(PetscSectionSetFieldName(section, 0, ""));
//   PetscCall(PetscSectionSetComponentName(section, 0, 0, "Pressure"));
//   PetscCall(PetscSectionSetComponentName(section, 0, 1, "VelocityX"));
//   PetscCall(PetscSectionSetComponentName(section, 0, 2, "VelocityY"));
//   PetscCall(PetscSectionSetComponentName(section, 0, 3, "VelocityZ"));
//   PetscCall(PetscSectionSetComponentName(section, 0, 4, "Temperature"));
//   PetscFunctionReturn(PETSC_SUCCESS);
// }

int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, PETSC_NULLPTR);

  DM          dm;
  Vec      coords_loc;
  PetscInt coords_loc_size, coords_dim;

  DMPlexCreateFromFile(PETSC_COMM_WORLD, "test.cgns", "ex16_plex", PETSC_TRUE, &dm);

  PetscCall(DMGetCoordinatesLocal(dm, &coords_loc));
  PetscCall(DMGetCoordinateDim(dm, &coords_dim));
  PetscCall(VecGetLocalSize(coords_loc, &coords_loc_size));

  auto num_local_nodes = coords_loc_size / coords_dim;
  std::cout << num_local_nodes << std::endl;

  vtkNew<vtkUnstructuredGrid> grid;
  vtkNew<vtkDoubleArray> pts_array;
  pts_array->SetNumberOfComponents(3);
//  pts_array->SetNumberOfTuples(num_local_nodes);
  double* pts_ptr;
  PetscCall(VecGetArray(coords_loc, &pts_ptr));
  pts_array->SetArray(pts_ptr, coords_loc_size, 1);
  vtkNew<vtkPoints> pts;
  pts->SetData(pts_array.GetPointer());
  grid->SetPoints(pts.GetPointer());

  PetscInt closure_dof, *closure_indices, elem_size;

  std::cout << dm->localSection << std::endl;

  DM cdm;
  PetscCall(DMGetCoordinateDM(dm, &cdm));

  std::cout << cdm->localSection << std::endl;

  PetscInt          cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  std::cout << ">>>>" << cStart << " " << cEnd << std::endl;

  DMPolytopeType cell_type;
  PetscCall(DMPlexGetCellType(dm, 0, &cell_type));
  PetscCall(DMPlexGetClosureIndices(cdm, cdm->localSection, cdm->localSection, 0, PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
  const int *perm;
  PetscCall(DMPlexCGNSGetPermutation_Internal(cell_type, closure_dof / coords_dim, &perm));
  std::cout << cell_type << " " << DM_POLYTOPE_HEXAHEDRON << " " << closure_dof << std::endl;
  for(int i=0; i<24; i++)
  {
    std::cout << closure_indices[i] << std::endl;
    // std::cout << ": " << perm[i]*3 << std::endl;
    // std::cout << closure_indices[perm[i]*3] << std::endl;
  }

  // PetscInt vStart, vEnd;
  // DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
  // PetscInt *closure;
  // PetscInt numPoints;
  // PetscCall(DMPlexGetTransitiveClosure(dm, 0, PETSC_TRUE, &numPoints, &closure));
  // std::cout << numPoints << std::endl;
  // std::cout << vStart << " " << vEnd << std::endl;

  // for (PetscInt i = 0; i < numPoints * 2; i += 2) {
  //     PetscInt point = closure[i] - vStart;
  //     std::cout << point << std::endl;
  // }

  // vtkNew<vtkXMLUnstructuredGridWriter> writer;
  // writer->SetInputData(grid.GetPointer());
  // writer->SetFileName("grid.vtu");
  // writer->SetDataModeToAscii();
  // writer->Write();
/*
  Vec         V;
  PetscViewer viewer;
  const char *name;
  PetscReal   time;
  PetscBool   set;
  comm = PETSC_COMM_WORLD;

  // Load DM from CGNS file
  PetscCall(ReadCGNSDM(comm, infilename, &dm));
  PetscCall(DMSetOptionsPrefix(dm, "loaded_"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  // Load solution from CGNS file
  PetscCall(PetscViewerCGNSOpen(comm, infilename, FILE_MODE_READ, &viewer));
  PetscCall(DMGetGlobalVector(dm, &V));
  PetscCall(PetscViewerCGNSSetSolutionIndex(viewer, 1));
  PetscCall(PetscViewerCGNSGetSolutionName(viewer, &name));
  PetscCall(PetscViewerCGNSGetSolutionTime(viewer, &time, &set));
  PetscCall(PetscPrintf(comm, "Solution Name: %s, and time %g\n", name, (double)time));
  PetscCall(VecLoad(V, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // Write loaded solution to CGNS file
  PetscCall(PetscViewerCGNSOpen(comm, user.outfile, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(V, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(DMRestoreGlobalVector(dm, &V));
  */
  //DMDestroy(&dm);

  PetscFinalize();
  return 0;
}

/*TEST
  build:
    requires: cgns
  testset:
    suffix: cgns
    requires: !complex
    nsize: 4
    args: -infile ${wPETSC_DIR}/share/petsc/datafiles/meshes/2x2x2_Q3_wave.cgns -outfile 2x2x2_Q3_wave_output.cgns
    args: -dm_plex_cgns_parallel -loaded_dm_view
    test:
      suffix: simple
      args: -petscpartitioner_type simple
TEST*/
