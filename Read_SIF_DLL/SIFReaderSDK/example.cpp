//---------------------------------------------------------------------------

#pragma hdrstop

#include "ATSIFIO.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

//---------------------------------------------------------------------------

void retrieveErrorCode(unsigned int _ui_ret, char * sz_error);
void retrievePropertyType(ATSIF_PropertyType _propType, char * sz_propertyType);
int getUserInput();

#pragma argsused
int main(int argc, char* argv[])
{
  AT_32 at32_userInput;
  AT_U32 atu32_ret, atu32_noFrames, atu32_frameSize, atu32_noSubImages;
  char *  sz_fileName = new char[MAX_PATH];
  char *  sz_error = new char[MAX_PATH];
  float * imageBuffer;
  size_t strLength;
  memset(sz_fileName, '\0', MAX_PATH);
  memset(sz_error, '\0', MAX_PATH);

  at32_userInput = getUserInput();
  while(at32_userInput != 0) {
    if(at32_userInput == 1) {
      atu32_ret = ATSIF_SetFileAccessMode(ATSIF_ReadAll);
      if(atu32_ret != ATSIF_SUCCESS) {
        printf("Could not set File access Mode. Error: %u\n", atu32_ret);
      }
      else {
        printf("File to open : ");
        fflush(stdin);
        fgets(sz_fileName, MAX_PATH, stdin);
        strLength = strlen(sz_fileName);
        sz_fileName[strLength - 1] = '\0';
        atu32_ret = ATSIF_ReadFromFile(sz_fileName);
        if(atu32_ret != ATSIF_SUCCESS) {
          printf("Could not open File : %s.\nError: %u\n", sz_fileName,atu32_ret);
        }
        else {
          atu32_ret = ATSIF_GetNumberFrames(ATSIF_Signal, &atu32_noFrames);
          if(atu32_ret != ATSIF_SUCCESS) {
            printf("Could not Get Number Frames. Error: %u\n", atu32_ret);
          }
          else {
            printf("Image contains %u frames.\n", atu32_noFrames);
            atu32_ret = ATSIF_GetFrameSize(ATSIF_Signal, &atu32_frameSize);
            if(atu32_ret != ATSIF_SUCCESS) {
              printf("Could not Get Frame Size. Error: %u\n", atu32_ret);
            }
            else {
              printf("Each frame contains %u pixels.\n", atu32_frameSize);
              atu32_ret = ATSIF_GetNumberSubImages(ATSIF_Signal, &atu32_noSubImages);
              if(atu32_ret != ATSIF_SUCCESS) {
                printf("Could not Get Number Sub Images. Error: %u\n", atu32_ret);
              }
              }printf("Each frame contains %u sub images.\n", atu32_noSubImages);
              for(AT_U32 i = 0; i < atu32_noSubImages; ++i) {
                AT_U32 atu32_left,atu32_bottom,atu32_right,atu32_top,atu32_hBin, atu32_vBin;
                printf("SubImage %u Properties:\n", (i + 1));
                atu32_ret = ATSIF_GetSubImageInfo(ATSIF_Signal,
                                                  i,
                                                  &atu32_left,&atu32_bottom,
                                                  &atu32_right,&atu32_top,
                                                  &atu32_hBin,&atu32_vBin);
                if(atu32_ret != ATSIF_SUCCESS) {
                  printf("Could not Get Sub Image Info. Error: %u\n", atu32_ret);
                }
                else {
                  printf("\tleft\t: %u\tbottom\t: %u\n", atu32_left, atu32_bottom);
                  printf("\tright\t: %u\ttop\t: %u\n", atu32_right, atu32_top);
                  printf("\thBin\t: %u\tvBin\t: %u\n", atu32_hBin, atu32_vBin);
                }
              }
              imageBuffer  = new float[atu32_frameSize];
              memset(imageBuffer, 0, atu32_frameSize);


              atu32_ret = ATSIF_GetFrame(ATSIF_Signal,0, imageBuffer, atu32_frameSize);
              if(atu32_ret != ATSIF_SUCCESS) {
                printf("Could not Get Frame. Error: %u\n", atu32_ret);
              }
              else {
                printf("The first 20 pixel values are : \n");
                for(int i = 0; i < 20; ++i) {
                  printf("%f\n", imageBuffer[i]);
                }
              }
              delete [] imageBuffer;
          }
        }
      }

    }
    else if(at32_userInput == 2) {
      char *  sz_propertyName = new char[MAX_PATH];
      memset(sz_propertyName, '\0', MAX_PATH);
      char *  sz_propertyValue = new char[MAX_PATH];
      memset(sz_propertyValue, '\0', MAX_PATH);
      char *  sz_propertyType = new char[MAX_PATH];
      memset(sz_propertyType, '\0', MAX_PATH);

      atu32_ret = ATSIF_SetFileAccessMode(ATSIF_ReadHeaderOnly);
      if(atu32_ret != ATSIF_SUCCESS) {
        printf("Could not set File access Mode. Error: %u\n", atu32_ret);
      }
      else {
        printf("File to open : ");
        fflush(stdin);
        fgets(sz_fileName, MAX_PATH, stdin);
        strLength = strlen(sz_fileName);
        sz_fileName[strLength - 1] = '\0';
        atu32_ret = ATSIF_ReadFromFile(sz_fileName);
        if(atu32_ret != ATSIF_SUCCESS) {
          printf("Could not open File : %s.\nError: %u\n", sz_fileName,atu32_ret);
        }
        printf("Property : ");
        fflush(stdin);
        fgets(sz_propertyName, MAX_PATH, stdin);
        strLength = strlen(sz_propertyName);
        sz_propertyName[strLength - 1] = '\0';
        atu32_ret = ATSIF_GetPropertyValue(ATSIF_Signal, sz_propertyName, sz_propertyValue, MAX_PATH);
        if(atu32_ret != ATSIF_SUCCESS) {
          retrieveErrorCode(atu32_ret, sz_error);
          printf("Could not get Property Value.\nError code : %u\nError Mesage : %s", atu32_ret, sz_error);
        }
        else {
          printf("Property Value : %s\n", sz_propertyValue);
        }
        ATSIF_PropertyType pType;
        atu32_ret = ATSIF_GetPropertyType(ATSIF_Signal, sz_propertyName, &pType);
        if(atu32_ret != ATSIF_SUCCESS) {
          retrieveErrorCode(atu32_ret, sz_error);
          printf("Could not get Property Type.\nError code : %u\nError Mesage : %s", atu32_ret, sz_error);
        }
        else {
          retrievePropertyType(pType, sz_propertyType);
          printf("Property Value : %s\n", sz_propertyType);
        }
      }
    }
    else {
      printf("Invalid option. Try Again!\n");
    }
    at32_userInput = getUserInput();
  }
  delete [] sz_fileName;
  system("PAUSE");
        return 0;
}
//---------------------------------------------------------------------------

void retrieveErrorCode(unsigned int _ui_ret, char * sz_error) {

  int paramNotValid(0);
  switch(_ui_ret) {
    case(ATSIF_SIF_FORMAT_ERROR):
      _snprintf(sz_error, MAX_PATH, "SIF FORMAT ERROR");
      break;
    case(ATSIF_NO_SIF_LOADED):
      _snprintf(sz_error, MAX_PATH, "SIF NOT LOADED");
      break;
    case(ATSIF_FILE_NOT_FOUND):
      _snprintf(sz_error, MAX_PATH, "FILE NOT FOUND");
      break;
    case(ATSIF_FILE_ACCESS_ERROR):
      _snprintf(sz_error, MAX_PATH, "FILE ACCESS ERROR");
      break;
    case(ATSIF_DATA_NOT_PRESENT):
      _snprintf(sz_error, MAX_PATH, "SIF DATA NOT PRESENT");
      break;
    case(ATSIF_P8INVALID):
      ++paramNotValid;
    case(ATSIF_P7INVALID):
      ++paramNotValid;
    case(ATSIF_P6INVALID):
      ++paramNotValid;
    case(ATSIF_P5INVALID):
      ++paramNotValid;
    case(ATSIF_P4INVALID):
      ++paramNotValid;
    case(ATSIF_P3INVALID):
      ++paramNotValid;
    case(ATSIF_P2INVALID):
      ++paramNotValid;
    case(ATSIF_P1INVALID):
      ++paramNotValid;
      _snprintf(sz_error, MAX_PATH, "PARAMETER %i NOT VALID", paramNotValid);
      break;
    default:
      sz_error = "UNKNOWN ERROR";
      break;
  }
}

void retrievePropertyType(ATSIF_PropertyType _propType, char * sz_propertyType) {

  switch(_propType) {
    case(ATSIF_AT_8):
      _snprintf(sz_propertyType, MAX_PATH, "char");
      break;
    case(ATSIF_AT_U8):
      _snprintf(sz_propertyType, MAX_PATH, "unsigned char");
      break;
    case(ATSIF_AT_32):
      _snprintf(sz_propertyType, MAX_PATH, "bool/int/long");
      break;
    case(ATSIF_AT_U32):
      _snprintf(sz_propertyType, MAX_PATH, "unsigned int/unsigned long");
      break;
    case(ATSIF_Float):
      _snprintf(sz_propertyType, MAX_PATH, "float");
      break;
    case(ATSIF_Double):
      _snprintf(sz_propertyType, MAX_PATH, "double");
      break;
    case(ATSIF_String):
      _snprintf(sz_propertyType, MAX_PATH, "char * ");
      break;
    default:
      _snprintf(sz_propertyType, MAX_PATH, "unknown");
      break;
  }
}

int getUserInput() {

  AT_32 choice;
  printf("\nSIFIO Example Menu  : \n\n");
  printf("\t1. Test File load.\n");
  printf("\t2. Test Properties.\n");
  printf("\t0. Exit.\n\n");

  scanf("%i", &choice);
  return choice;
}

