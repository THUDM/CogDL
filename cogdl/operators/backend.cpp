#include <iostream>
#include <Python.h>
using namespace std;
int backend(){
    Py_Initialize();
    string b1="torch";
    PyObject * pModule = NULL;
    PyObject * pFunc = NULL;
    pModule = PyImport_ImportModule("cogdl.backend");
    pFunc = PyObject_GetAttrString(pModule, "backend"); 
    PyObject* pResult = PyObject_CallObject(pFunc, NULL); 
    string cogdl_backend = PyUnicode_AsUTF8(pResult);
    int flag;
    flag=b1.compare(cogdl_backend);
    Py_Finalize();
    return (flag);
}