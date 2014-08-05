#include <Python.h>

typedef struct {
    PyObject_HEAD
    // TODO
} trainer_ModelObject;

static PyTypeObject trainer_ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* tp_name */        "trainer.Model",
    /* tp_basicsize */   sizeof(trainer_ModelObject),
    /* tp_itemsize */    0,
    /* tp_dealloc */     0,
    /* tp_print */       0,
    /* tp_getattr */     0,
    /* tp_setattr */     0,
    /* tp_reserved */    0,
    /* tp_repr */        0,
    /* tp_as_number */   0,
    /* tp_as_sequence */ 0,
    /* tp_as_mapping */  0,
    /* tp_hash  */       0,
    /* tp_call */        0,
    /* tp_str */         0,
    /* tp_getattro */    0,
    /* tp_setattro */    0,
    /* tp_as_buffer */   0,
    /* tp_flags */       Py_TPFLAGS_DEFAULT,
    /* tp_doc */         "Collaborative filtering trainer model.",
};

static PyModuleDef trainermodule = {
    PyModuleDef_HEAD_INIT,
    "trainer",
    "Collaborative filtering trainer module.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_trainer(void) 
{
    PyObject* module;

    trainer_ModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&trainer_ModelType) < 0) {
        return NULL;
    }

    module = PyModule_Create(&trainermodule);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&trainer_ModelType);
    PyModule_AddObject(module, "Model", (PyObject*)&trainer_ModelType);

    return module;
}
