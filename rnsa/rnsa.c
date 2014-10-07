#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    /* Row count. */               unsigned long row_count;
    /* Column count. */            unsigned long column_count;
    /* Value count. */             unsigned long value_count;
    /* Cluster count. */           unsigned long k;
    /* Points to column starts. */ unsigned long *indptr;
    /* Row indices. */             unsigned long *indices;
    /* Corresponding values. */    float *values;
    /* Cluster centers. */         float *centroids;
} Model;

/*
  Model initialization and deallocation.
--------------------------------------------------------------------------------
 */

static PyObject *
model_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Model *self = (Model*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->column_count = self->value_count = self->k = 0;
        self->indices = self->indptr = NULL;
        self->values = NULL;
        self->centroids = NULL;
    }
    return (PyObject*)self;
}

static int
model_init(Model *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"row_count", "column_count", "value_count", "k", NULL};
    if (!PyArg_ParseTupleAndKeywords(
        args, kwargs, "kkkk", kwlist,
        &self->row_count, &self->column_count, &self->value_count, &self->k
    )) {
        return -1;
    }
    if (
        !(self->indptr = PyMem_RawMalloc(self->column_count * sizeof(*self->indptr) + sizeof(*self->indptr))) ||
        !(self->indices = PyMem_RawMalloc(self->value_count * sizeof(*self->indices))) ||
        !(self->values = PyMem_RawMalloc(self->value_count * sizeof(*self->values))) ||
        !(self->centroids = PyMem_RawMalloc(self->k * self->row_count * sizeof(*self->centroids)))
    ) {
        PyErr_NoMemory();
        return -1;
    }
    self->indptr[0] = 0ul;
    self->indptr[self->column_count] = self->value_count;
    return 0;
}

static void
model_dealloc(Model *self) {
    PyMem_RawFree(self->indptr);
    PyMem_RawFree(self->indices);
    PyMem_RawFree(self->values);
    PyMem_RawFree(self->centroids);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
  Model setters.
--------------------------------------------------------------------------------
 */

static PyObject *
model_set_indptr(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long j, index;

    static char *kwlist[] = {"j", "index", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kk", kwlist, &j, &index)) {
        return NULL;
    }

    if ((j == 0) || (j >= self->column_count)) {
        PyErr_SetObject(PyExc_ValueError, Py_BuildValue("k", j));
        return NULL;
    }

    self->indptr[j] = index;

    Py_RETURN_NONE;
}

static PyObject *
model_set_value(Model *self, PyObject *args, PyObject *kwargs) {
    unsigned long index;
    float value;

    static char *kwlist[] = {"index", "value", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "kf", kwlist, &index, &value)) {
        return NULL;
    }

    if (index >= self->value_count) {
        PyErr_SetObject(PyExc_ValueError, Py_BuildValue("k", index));
        return NULL;
    }

    self->values[index] = value;

    Py_RETURN_NONE;
}

/*
  Model definition.
--------------------------------------------------------------------------------
 */

static PyMemberDef model_members[] = {
    {"row_count", T_ULONG, offsetof(Model, row_count), 0, "Row count."},
    {"column_count", T_ULONG, offsetof(Model, column_count), 0, "Column count."},
    {"value_count", T_ULONG, offsetof(Model, value_count), 0, "Value count."},
    {"k", T_ULONG, offsetof(Model, k), 0, "Cluster count."},
    {NULL}
};

static PyMethodDef model_methods[] = {
    {"set_indptr", (PyCFunction)model_set_indptr, METH_VARARGS | METH_KEYWORDS, "Sets column start index."},
    {"set_value", (PyCFunction)model_set_value, METH_VARARGS | METH_KEYWORDS, "Sets value at the specified row position."},
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

    module = PyModule_Create(&rnsa_module);
    if (module == NULL) {
        return NULL;
    }

    ModelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ModelType) < 0) {
        return NULL;
    }

    Py_INCREF(&ModelType);
    PyModule_AddObject(module, "Model", (PyObject*)&ModelType);

    return module;
}
