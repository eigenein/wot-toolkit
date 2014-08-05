#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    int row_count;
    int column_count;
    int value_count;
    int *rows;
    int *columns;
    double *values;
} Model;

static PyObject *
model_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Model *self = (Model*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->row_count = 0;
        self->column_count = 0;
        self->value_count = 0;
        self->rows = NULL;
        self->columns = NULL;
        self->values = NULL;
    }
    return (PyObject*)self;
}

int alloc_wrapper(size_t n, void **p) {
    *p = PyMem_RawMalloc(n);
    if (*p != NULL) {
        return 1;
    } else {
        PyErr_SetString(PyExc_MemoryError, "not enough memory");
        return 0;
    }
}

static int
model_init(Model *self, PyObject *args, PyObject *kwargs) {
    // Parse arguments.
    static char *kwlist[] = {"row_count", "column_count", "value_count", "data", NULL};
    PyObject *data;
    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "iiiO",
        kwlist,
        &self->row_count,
        &self->column_count,
        &self->value_count,
        &data
    )) {
        return -1;
    }
    // Allocate memory.
    if (
        !alloc_wrapper(self->row_count * sizeof(int), (void**)&self->rows) ||
        !alloc_wrapper(self->column_count * sizeof(int), (void**)&self->columns) ||
        !alloc_wrapper(self->value_count * sizeof(double), (void**)&self->values)
    ) {
        return -1;
    }
    return 0;
}

static void
model_dealloc(Model *self) {
    PyMem_RawFree(self->rows);
    PyMem_RawFree(self->columns);
    PyMem_RawFree(self->values);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyMemberDef model_members[] = {
    {"row_count", T_INT, offsetof(Model, row_count), 0, "Row count."},
    {"column_count", T_INT, offsetof(Model, column_count), 0, "Column count."},
    {NULL}
};

static PyMethodDef model_methods[] = {
    {NULL}
};

static PyTypeObject ModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* tp_name */          "trainer.Model",
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
    /* tp_doc */            "Collaborative filtering trainer model.",
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
    PyObject *module;

    ModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ModelType) < 0) {
        return NULL;
    }

    module = PyModule_Create(&trainermodule);
    if (module == NULL) {
        return NULL;
    }

    Py_INCREF(&ModelType);
    PyModule_AddObject(module, "Model", (PyObject*)&ModelType);

    return module;
}
