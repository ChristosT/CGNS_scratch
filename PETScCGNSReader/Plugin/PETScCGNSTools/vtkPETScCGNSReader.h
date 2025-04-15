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

class PETSCCGNSTOOLS_EXPORT vtkPETScCGNSReader : public vtkUnstructuredGridAlgorithm
{
public:
  static vtkPETScCGNSReader* New();
  vtkTypeMacro(vtkPETScCGNSReader, vtkUnstructuredGridAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetStdStringFromCharMacro(FileName);
  vtkGetCharFromStdStringMacro(FileName);

protected:
  std::string FileName;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  vtkPETScCGNSReader();
  ~vtkPETScCGNSReader() override;

private:
  vtkPETScCGNSReader(const vtkPETScCGNSReader&) = delete;
  void operator=(const vtkPETScCGNSReader&) = delete;

  class vtkInternals;
  std::unique_ptr<vtkInternals> Internals;
};

#endif
