#include <Python.h>
#include <structmember.h>

/*
  Module definition.
--------------------------------------------------------------------------------
 */

static PyModuleDef rnsa_module = {
    PyModuleDef_HEAD_INIT,
    "rnsa",
    "Refined Neighbor Selection Algorithm extension.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_rnsa(void) 
{
    PyObject *module;

    module = PyModule_Create(&rnsa_module);
    if (module == NULL) {
        return NULL;
    }

    return module;
}
