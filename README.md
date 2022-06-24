# This is a repository for PeakSelector software for PALM/iPALM data processing

## PeakSel_Docs directory contains extensive manuals on running the code
- PeakSelector_2D_Data_Processing_Analysis.doc   - explaingn how to use PeakSelector for 2D SMLM / PALM
- PeakSelector_3D_Data_Processing_Analysis.doc   - explaingn how to use PeakSelector for iPALM
- PeakSelector_3D_Data_Processing_Analysis_Astig_only.doc   - explaingn how to use PeakSelector for astigmatic 3D SMLM / PALM


Processing on a windows-based computer can be done under following IDL (version 6.4 or later) environments:
- IDL Virtual Machine (VM) environment
- IDL Run-Time (RT) license
- IDL Full Development (FD) license.

Get IDL software here:
https://www.l3harrisgeospatial.com/docs/licensingoptions.html


Fast processing, utilizing computer multi-treading capabilities (IDL bridge), RT or FD licenses are required, this cannot be done under VM license. 
Regular-sized iPALM data files can be processed on a Windows-based computer, but require high-end processors (ideally dual E5-2697 processors) and RT or FD licences.
To start PeakSelector under FD license:
- Start IDL.
- Select and load PeakSelector.prj project.
- Compile and Run the project.
