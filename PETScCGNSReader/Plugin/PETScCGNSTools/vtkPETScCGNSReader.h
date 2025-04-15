// SPDX-FileCopyrightText: Copyright (c) Ken Martin, Will Schroeder, Bill
// Lorensen SPDX-License-Identifier: BSD-3-Clause
/**
 * @class   vtkPETScCGNSReader
 * @brief   TODO
 *
 * TODO
 *
 */

#ifndef vtkPETScCGNSReader_h
#define vtkPETScCGNSReader_h
#include "PETScCGNSToolsModule.h" // for export macro
#include "vtkUnstructuredGridAlgorithm.h"

#include <memory>

class vtkDoubleArray;
class vtkMultiProcessController;

class PETSCCGNSTOOLS_EXPORT vtkPETScCGNSReader : public vtkUnstructuredGridAlgorithm
{
public:
  static vtkPETScCGNSReader* New();
  vtkTypeMacro(vtkPETScCGNSReader, vtkUnstructuredGridAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetStdStringFromCharMacro(FileName);
  vtkGetCharFromStdStringMacro(FileName);

  ///@{
  /**
   * Get/Set the parallel controller to use. By default, set to.
   * `vtkMultiProcessController::GlobalController`.
   */
  void SetController(vtkMultiProcessController* controller);
  vtkGetObjectMacro(Controller, vtkMultiProcessController);
  ///@}

protected:
  std::string FileName;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int RequestInformation(vtkInformation* request, vtkInformationVector** inputVector,
    vtkInformationVector* outputVector) override;

  vtkPETScCGNSReader();
  ~vtkPETScCGNSReader() override;

  vtkMultiProcessController* Controller;

private:
  vtkPETScCGNSReader(const vtkPETScCGNSReader&) = delete;
  void operator=(const vtkPETScCGNSReader&) = delete;

  class vtkInternals;
  std::unique_ptr<vtkInternals> Internals;
};

#endif
