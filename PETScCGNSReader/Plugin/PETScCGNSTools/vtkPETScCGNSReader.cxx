#include "petscsf.h"

#include "petsc/private/dmpleximpl.h" /*I   "petscdmplex.h"   I*/
#include "petscdmplex.h"

#include "vtkDoubleArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMPI.h"
#include "vtkMPICommunicator.h"
#include "vtkMPIController.h"
#include "vtkMultiProcessController.h"
#include "vtkNew.h"
#include "vtkObjectFactory.h"
#include "vtkPETScCGNSReader.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkUnstructuredGrid.h"
#include "vtkUnstructuredGridAlgorithm.h"

#if !defined(CGNS_ENUMT)
#define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
#define CGNS_ENUMV(a) a
#endif

// PetscCall returns 0 oin success which is opposite to VTK convention. So we create this wrapper
#define VTKPetscCall(...)                                                                          \
  do                                                                                               \
  {                                                                                                \
    PetscErrorCode ierr_petsc_call_q_;                                                             \
    PetscStackUpdateLine;                                                                          \
    ierr_petsc_call_q_ = __VA_ARGS__;                                                              \
    if (PetscUnlikely(ierr_petsc_call_q_ != PETSC_SUCCESS))                                        \
    {                                                                                              \
      (void)PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__,                   \
        ierr_petsc_call_q_, PETSC_ERROR_REPEAT, " ");                                              \
      return EXIT_FAILURE;                                                                         \
    }                                                                                              \
  } while (0)

#define VTKPetscCallNoReturnValue(...)                                                             \
  do                                                                                               \
  {                                                                                                \
    PetscErrorCode ierr_petsc_call_q_;                                                             \
    PetscStackUpdateLine;                                                                          \
    ierr_petsc_call_q_ = __VA_ARGS__;                                                              \
    if (PetscUnlikely(ierr_petsc_call_q_ != PETSC_SUCCESS))                                        \
    {                                                                                              \
      (void)PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__,                   \
        ierr_petsc_call_q_, PETSC_ERROR_REPEAT, " ");                                              \
    }                                                                                              \
  } while (0)

// Permute plex closure ordering to CGNS
static PetscErrorCode DMPlexCGNSGetPermutation_Internal(
  DMPolytopeType cell_type, PetscInt closure_size, const int** perm)
{
  // https://cgns.github.io/CGNS_docs_current/sids/conv.html#unstructgrid
  static const int bar_2[2] = { 0, 1 };
  static const int bar_3[3] = { 1, 2, 0 };
  static const int bar_4[4] = { 2, 3, 0, 1 };
  static const int bar_5[5] = { 3, 4, 0, 1, 2 };
  static const int tri_3[3] = { 0, 1, 2 };
  static const int tri_6[6] = { 3, 4, 5, 0, 1, 2 };
  static const int tri_10[10] = { 7, 8, 9, 1, 2, 3, 4, 5, 6, 0 };
  static const int quad_4[4] = { 0, 1, 2, 3 };
  static const int quad_9[9] = {
    5, 6, 7, 8, // vertices
    1, 2, 3, 4, // edges
    0,          // center
  };
  static const int quad_16[] = {
    12, 13, 14, 15,           // vertices
    4, 5, 6, 7, 8, 9, 10, 11, // edges
    0, 1, 3, 2,               // centers
  };
  static const int quad_25[] = {
    21, 22, 23, 24,                                // vertices
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, // edges
    0, 1, 2, 5, 8, 7, 6, 3, 4,                     // centers
  };
  static const int tetra_4[4] = { 0, 2, 1, 3 };
  static const int tetra_10[10] = { 6, 8, 7, 9, 2, 1, 0, 3, 5, 4 };
  static const int tetra_20[20] = {
    16, 18, 17, 19,         // vertices
    9, 8, 7, 6, 5, 4,       // bottom edges
    10, 11, 14, 15, 13, 12, // side edges
    0, 2, 3, 1,             // faces
  };
  static const int hexa_8[8] = { 0, 3, 2, 1, 4, 5, 6, 7 };
  static const int hexa_27[27] = {
    19, 22, 21, 20, 23, 24, 25, 26, // vertices
    10, 9, 8, 7,                    // bottom edges
    16, 15, 18, 17,                 // mid edges
    11, 12, 13, 14,                 // top edges
    1, 3, 5, 4, 6, 2,               // faces
    0,                              // center
  };
  static const int hexa_64[64] = {
    // debug with $PETSC_ARCH/tests/dm/impls/plex/tests/ex49 -dm_plex_simplex
    // 0 -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -dm_coord_petscspace_degree 3
    56, 59, 58, 57, 60, 61, 62, 63, // vertices
    39, 38, 37, 36, 35, 34, 33, 32, // bottom edges
    51, 50, 48, 49, 52, 53, 55,
    54, // mid edges; Paraview needs edge 21-22 swapped with 23-24
    //    51, 50, 48, 49, 53, 52, 54, 55, // mid edges; Paraview needs edge
    //    21-22 swapped with 23-24
    40, 41, 42, 43, 44, 45, 46, 47, // top edges
    8, 10, 11, 9,                   // z-minus face
    16, 17, 19, 18,                 // y-minus face
    24, 25, 27, 26,                 // x-plus face
    20, 21, 23, 22,                 // y-plus face
    30, 28, 29, 31,                 // x-minus face
    12, 13, 15, 14,                 // z-plus face
    0, 1, 3, 2, 4, 5, 7, 6,         // center
  };

  PetscFunctionBegin;
  *perm = NULL;
  switch (cell_type)
  {
    case DM_POLYTOPE_SEGMENT:
      switch (closure_size)
      {
        case 2:
          *perm = bar_2;
        case 3:
          *perm = bar_3;
        case 4:
          *perm = bar_4;
          break;
        case 5:
          *perm = bar_5;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT,
            DMPolytopeTypes[cell_type], closure_size);
      }
      break;
    case DM_POLYTOPE_TRIANGLE:
      switch (closure_size)
      {
        case 3:
          *perm = tri_3;
          break;
        case 6:
          *perm = tri_6;
          break;
        case 10:
          *perm = tri_10;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT,
            DMPolytopeTypes[cell_type], closure_size);
      }
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      switch (closure_size)
      {
        case 4:
          *perm = quad_4;
          break;
        case 9:
          *perm = quad_9;
          break;
        case 16:
          *perm = quad_16;
          break;
        case 25:
          *perm = quad_25;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT,
            DMPolytopeTypes[cell_type], closure_size);
      }
      break;
    case DM_POLYTOPE_TETRAHEDRON:
      switch (closure_size)
      {
        case 4:
          *perm = tetra_4;
          break;
        case 10:
          *perm = tetra_10;
          break;
        case 20:
          *perm = tetra_20;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT,
            DMPolytopeTypes[cell_type], closure_size);
      }
      break;
    case DM_POLYTOPE_HEXAHEDRON:
      switch (closure_size)
      {
        case 8:
          *perm = hexa_8;
          break;
        case 27:
          *perm = hexa_27;
          break;
        case 64:
          *perm = hexa_64;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT,
            DMPolytopeTypes[cell_type], closure_size);
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT,
        DMPolytopeTypes[cell_type], closure_size);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

class vtkPETScCGNSReader::vtkInternals
{
public:
  // hold pointers to objects we need to clear
  PetscViewer viewer;
  Vec local_sln, global_sln;
  PetscScalar* slnArray = { nullptr };
  DM* dm = { nullptr };
  vtkPETScCGNSReader* Parent;

  static int petsc_schwarz_counter;

  void Clear()
  {
    PetscViewerDestroy(&this->viewer);
    VecRestoreArray(this->local_sln, &this->slnArray);
    if (this->dm)
    {
      DMRestoreLocalVector(*dm, &local_sln);
      DMRestoreGlobalVector(*dm, &global_sln);
    }
    this->dm = nullptr;
  }

  void Initialize()
  {
    if (this->petsc_schwarz_counter == 0)
    {
      // PETSC_COMM_WORLD needs to be set before PetscInitializeNoArguments so
      // that it uses the proper communicator
      if (vtkMPIController* mpiController =
            vtkMPIController::SafeDownCast(this->Parent->GetController()))
      {
        vtkMPICommunicator* mpiCommunicator =
          vtkMPICommunicator::SafeDownCast(mpiController->GetCommunicator());
        assert(mpiCommunicator);
        PETSC_COMM_WORLD = *mpiCommunicator->GetMPIComm()->GetHandle();
      }
      PetscOptionsSetValue(NULL, "-dm_plex_cgns_parallel", "1");
      PetscInitializeNoArguments();
    }
    this->petsc_schwarz_counter++;
  }

  vtkInternals(vtkPETScCGNSReader* parent) { this->Parent = parent; }

  ~vtkInternals()
  {
    this->petsc_schwarz_counter--;
    if (this->petsc_schwarz_counter == 0)
    {
      PetscFinalize();
    }
  }

  vtkSmartPointer<vtkDoubleArray> LoadSolution(const char* fileName, DM* dm)
  {
    MPI_Comm comm = PETSC_COMM_WORLD;
    this->dm = dm;

    PetscReal time;
    PetscBool set;

    // Set section component names, used when writing out CGNS files
    PetscSection section;
    DMGetLocalSection(*dm, &section);

    PetscSectionSetFieldName(section, 0, "");
    PetscSectionSetComponentName(section, 0, 0, "Pressure");
    PetscSectionSetComponentName(section, 0, 1, "VelocityX");
    PetscSectionSetComponentName(section, 0, 2, "VelocityY");
    PetscSectionSetComponentName(section, 0, 3, "VelocityZ");
    PetscSectionSetComponentName(section, 0, 4, "Temperature");

    // Load solution from CGNS file
    PetscViewerCGNSOpen(comm, fileName, FILE_MODE_READ, &viewer);
    DMGetGlobalVector(*dm, &global_sln);
    PetscViewerCGNSSetSolutionIndex(viewer, -1);
    PetscInt idx;
    // PetscViewerCGNSGetSolutionName(viewer, &name);
    PetscViewerCGNSGetSolutionTime(viewer, &time, &set);
    VecLoad(global_sln, viewer);

    DMGetLocalVector(*dm, &local_sln);

    // Transfer data from global vector to local vector (with ghost points)
    DMGlobalToLocalBegin(*dm, global_sln, INSERT_VALUES, local_sln);
    DMGlobalToLocalEnd(*dm, global_sln, INSERT_VALUES, local_sln);

    PetscInt lsize;
    VecGetLocalSize(local_sln, &lsize);

    auto fields = vtkSmartPointer<vtkDoubleArray>::New();
    fields->SetName("fields");
    fields->SetNumberOfComponents(5);
    fields->SetNumberOfTuples(lsize / 5);
    VecGetArray(local_sln, &slnArray);
    memcpy(fields->GetPointer(0), slnArray, lsize * sizeof(double));
    this->Clear();

    return fields;
  }
};
int vtkPETScCGNSReader::vtkInternals::petsc_schwarz_counter = 0;

vtkStandardNewMacro(vtkPETScCGNSReader);

vtkPETScCGNSReader::vtkPETScCGNSReader()
  : Controller(nullptr)
{
  this->SetNumberOfInputPorts(0);
  this->SetNumberOfOutputPorts(1);
  this->Internals = std::make_unique<vtkPETScCGNSReader::vtkInternals>(this);
  this->SetController(vtkMultiProcessController::GetGlobalController());
}
vtkCxxSetObjectMacro(vtkPETScCGNSReader, Controller, vtkMultiProcessController);

vtkPETScCGNSReader::~vtkPETScCGNSReader()
{
  this->SetController(nullptr);
}

void vtkPETScCGNSReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "FileName:" << this->FileName << endl;
  os << indent << "Controller: " << this->Controller << endl;
}

int vtkPETScCGNSReader::RequestInformation(
  vtkInformation* request, vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  outputVector->GetInformationObject(0)->Set(CAN_HANDLE_PIECE_REQUEST(), 1);
  return 1;
}

int vtkPETScCGNSReader::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
{

  this->Internals->Initialize();

  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  // get the output
  vtkUnstructuredGrid* grid = vtkUnstructuredGrid::GetData(outInfo);
  MPI_Comm comm = PETSC_COMM_WORLD;
  int rank = this->Controller->GetLocalProcessId();
  int size = this->Controller->GetNumberOfProcesses();

  DM dm;
  Vec coords_loc, local_sln, V;
  PetscInt coords_loc_size, coords_dim;
  PetscInt degree, dim;
  PetscViewer viewer;
  const char* name;
  PetscReal time;
  PetscBool set;

  PetscErrorCode error;

  VTKPetscCall(
    DMPlexCreateFromFile(PETSC_COMM_WORLD, this->FileName.c_str(), "ex16_plex", PETSC_TRUE, &dm));

  VTKPetscCall(DMSetUp(dm));
  VTKPetscCall(DMSetFromOptions(dm));

  PetscFE fe_natural;
  PetscDualSpace dual_space_natural;

  VTKPetscCall(DMGetField(dm, 0, NULL, (PetscObject*)&fe_natural));
  VTKPetscCall(PetscFEGetDualSpace(fe_natural, &dual_space_natural));
  VTKPetscCall(PetscDualSpaceGetOrder(dual_space_natural, &degree));
  VTKPetscCall(DMClearFields(dm));
  VTKPetscCall(DMSetLocalSection(dm, NULL));

  VTKPetscCall(DMGetDimension(dm, &dim));

  PetscInt cStart, cEnd;
  VTKPetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  DMPolytopeType cell_type;

  PetscFE fe;
  VTKPetscCall(DMPlexGetCellType(dm, cStart, &cell_type));
  VTKPetscCall(
    PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 5, cell_type, degree, PETSC_DETERMINE, &fe));
  VTKPetscCall(PetscObjectSetName((PetscObject)fe, "FE for VecLoad"));
  VTKPetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  VTKPetscCall(DMCreateDS(dm));
  VTKPetscCall(PetscFEDestroy(&fe));

  VTKPetscCall(DMGetCoordinatesLocal(dm, &coords_loc));
  VTKPetscCall(DMGetCoordinateDim(dm, &coords_dim));
  VTKPetscCall(VecGetLocalSize(coords_loc, &coords_loc_size));

  auto num_local_nodes = coords_loc_size / coords_dim;

  vtkNew<vtkDoubleArray> pts_array;
  pts_array->SetNumberOfComponents(3);
  const double* pts_ptr;
  VTKPetscCall(VecGetArrayRead(coords_loc, &pts_ptr));
  double* pts_copy = new double[coords_loc_size];
  memcpy(pts_copy, pts_ptr, coords_loc_size * sizeof(double));
  pts_array->SetArray(pts_copy, coords_loc_size, 0, vtkAbstractArray::VTK_DATA_ARRAY_DELETE);
  VecRestoreArrayRead(coords_loc, &pts_ptr);
  vtkNew<vtkPoints> pts;
  pts->SetData(pts_array.GetPointer());
  grid->SetPoints(pts.GetPointer());

  PetscInt closure_dof, *closure_indices, elem_size;

  DM cdm;
  VTKPetscCall(DMGetCoordinateDM(dm, &cdm));

  VTKPetscCall(DMPlexGetClosureIndices(cdm, cdm->localSection, cdm->localSection, cStart,
    PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
  DMPlexRestoreClosureIndices(cdm, cdm->localSection, cdm->localSection, cStart, PETSC_FALSE,
    &closure_dof, &closure_indices, NULL, NULL);

  PetscInt cSize = closure_dof / coords_dim;
  PetscInt nCells = cEnd - cStart;

  vtkNew<vtkIdTypeArray> connectivity;
  connectivity->SetNumberOfTuples(cSize * nCells);

  vtkNew<vtkIdTypeArray> offsets;
  offsets->SetNumberOfTuples(nCells + 1);
  for (PetscInt i = 0; i < nCells; i++)
  {
    offsets->SetValue(i, i * cSize);
  }
  offsets->SetValue(nCells, nCells * cSize);

  vtkNew<vtkCellArray> cells;
  cells->SetData(offsets, connectivity);

  static const int HEXA_8_ToVTK[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  static const int HEXA_27_ToVTK[27] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12,
    13, 14, 15, 24, 22, 21, 23, 20, 25, 26 };
  static const int HEXA_64_ToVTK[64] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12, 15, 14, 24,
    25, 26, 27, 29, 28, 31, 30, 16, 17, 18, 19, 22, 23, 20, 21, 49, 48, 50, 51, 40, 41, 43, 42, 36,
    37, 39, 38, 45, 44, 46, 47, 32, 33, 35, 34, 52, 53, 55, 54, 56, 57, 59, 58, 60, 61, 63, 62 };

  const int* translator = nullptr;

  vtkNew<vtkUnsignedCharArray> cellTypes;
  cellTypes->SetNumberOfTuples(nCells);
  if (cSize == 8)
  {
    cellTypes->FillValue(VTK_HEXAHEDRON);
    translator = HEXA_8_ToVTK;
  }
  else if (cSize == 27)
  {
    cellTypes->FillValue(VTK_TRIQUADRATIC_HEXAHEDRON);
    translator = HEXA_27_ToVTK;
  }
  else if (cSize == 64)
  {
    cellTypes->FillValue(VTK_LAGRANGE_HEXAHEDRON);
    translator = HEXA_64_ToVTK;
  }
  else
  {
    std::cerr << "Cell type not supported." << std::endl;
    return 1;
  }

  grid->SetCells(cellTypes, cells);
  PetscInt tmpids[VTK_CELL_SIZE];
  for (PetscInt cid = cStart, cid0 = 0; cid < cEnd; cid++, cid0++)
  {
    VTKPetscCall(DMPlexGetClosureIndices(cdm, cdm->localSection, cdm->localSection, cid,
      PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
    const int* perm;
    VTKPetscCall(DMPlexCGNSGetPermutation_Internal(cell_type, cSize, &perm));
    for (int i = 0; i < cSize; i++)
    {
      tmpids[i] = closure_indices[perm[i] * coords_dim] / coords_dim;
    }
    for (int i = 0; i < cSize; i++)
    {
      connectivity->SetValue(cid0 * cSize + i, tmpids[translator[i]]);
    }
    DMPlexRestoreClosureIndices(cdm, cdm->localSection, cdm->localSection, 0, PETSC_FALSE,
      &closure_dof, &closure_indices, NULL, NULL);
  }

  vtkSmartPointer<vtkDoubleArray> fields =
    this->Internals->LoadSolution(this->FileName.c_str(), &dm);
  grid->GetPointData()->AddArray(fields);

  PetscCall(DMDestroy(&dm));

  return 1;
}
