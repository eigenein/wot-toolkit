#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    /* Column count. */            int column_count;
    /* Value count. */             int value_count;
    /* Points to column starts. */ int *indptr;
    /* Row indices. */             int *indices;
    /* Corresponding values. */    double *values;
} Model;

/*
  Model initialization and deallocation.
--------------------------------------------------------------------------------
 */

static PyObject *
model_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Model *self = (Model*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->column_count = self->value_count = 0;
        self->indices = self->indptr = NULL;
        self->values = NULL;
    }
    return (PyObject*)self;
}

static int
model_init(Model *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"column_count", "value_count", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii", kwlist, &self->column_count, &self->value_count)) {
        return -1;
    }
    if (
        !(self->indptr = PyMem_RawMalloc(self->column_count * sizeof(int) + sizeof(int))) ||
        !(self->indices = PyMem_RawMalloc(self->value_count * sizeof(int))) ||
        !(self->values = PyMem_RawMalloc(self->value_count * sizeof(double)))
    ) {
        PyErr_NoMemory();
        return -1;
    }
    self->indptr[self->column_count] = self->value_count;
    return 0;
}

static void
model_dealloc(Model *self) {
    PyMem_RawFree(self->indptr);
    PyMem_RawFree(self->indices);
    PyMem_RawFree(self->values);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
  Model definition.
--------------------------------------------------------------------------------
 */

static PyMemberDef model_members[] = {
    // TODO.
    {NULL}
};

static PyMethodDef model_methods[] = {
    // TODO.
    {NULL}
};

static PyTypeObject ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* tp_name */          "rnsa.Model",
    /* tp_basicsize */      sizeof(Model),
    /* tp_itemsize */       0,
    /* tp_dealloc */        (destructor)model_dealloc,
    /* tp_print */          0,
    /* tp_getattr */        0,
    /* tp_setattr */        0,
    /* tp_reserved */       0,
    /* tp_repr */           0,
    /* tp_as_number */      0,
    /* tp_as_sequence */    0,
    /* tp_as_mapping */     0,
    /* tp_hash  */          0,
    /* tp_call */           0,
    /* tp_str */            0,
    /* tp_getattro */       0,
    /* tp_setattro */       0,
    /* tp_as_buffer */      0,
    /* tp_flags */          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    /* tp_doc */            "RNSA model.",
    /* tp_traverse */       0,
    /* tp_clear */          0,
    /* tp_richcompare */    0,
    /* tp_weaklistoffset */ 0,
    /* tp_iter */           0,
    /* tp_iternext */       0,
    /* tp_methods */        model_methods,
    /* tp_members */        model_members,
    /* tp_getset */         0,
    /* tp_base */           0,
    /* tp_dict */           0,
    /* tp_descr_get */      0,
    /* tp_descr_set */      0,
    /* tp_dictoffset */     0,
    /* tp_init */           (initproc)model_init,
    /* tp_alloc */          0,
    /* tp_new */            model_new,
};

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

    ModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ModelType) < 0) {
        return NULL;
    }

    module = PyModule_Create(&rnsa_module);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&ModelType);
    PyModule_AddObject(module, "Model", (PyObject*)&ModelType);

    return module;
}
